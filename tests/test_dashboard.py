"""API reconciliation, empty-state and read-only provenance checks."""

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gradientclimb.dashboard import create_app
from gradientclimb.experiments import RunRecorder, run_synthetic


def test_empty_dashboard_is_truthful_and_assets_are_local(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.get("/")
    assert response.status_code == 200
    assert "GradientClimb" in response.text
    assert "cdn" not in response.text
    assert client.get("/dashboard.css").status_code == 200
    script = client.get("/dashboard.js")
    assert script.status_code == 200
    assert "Not measured" in script.text
    assert client.get("/api/runs").json() == []
    assert client.get("/api/runs?algorithm=ppo").json() == []
    assert client.get("/api/metrics").json() == []
    assert client.get("/api/summary").json()["wall_clock_seconds"] is None
    assert client.get("/api/analytics/resources").json()["rows"] == []


def test_dashboard_queries_canonical_records_and_filters(tmp_path):
    slow = run_synthetic(tmp_path, seed=1, gain=0.05, steps=4)
    fast = run_synthetic(tmp_path, seed=2, gain=0.4, steps=4)
    client = TestClient(create_app(tmp_path))
    runs = client.get("/api/runs").json()
    assert {row["run_id"] for row in runs} == {slow["run_id"], fast["run_id"]}
    assert all(isinstance(row["configuration"], dict) for row in runs)
    assert client.get("/api/runs?algorithm=missing").json() == []
    assert len(client.get("/api/runs?algorithm=scalar-gradient").json()) == 2
    metrics = client.get(f"/api/metrics?run_id={slow['run_id']}&name=quality").json()
    assert len(metrics) == 4
    assert {row["run_id"] for row in metrics} == {slow["run_id"]}
    assert all(row["name"] == "quality" for row in metrics)
    summary = client.get("/api/summary").json()
    assert summary["total_runs"] == 2
    assert summary["environment_steps"] == 8
    assert summary["metric_count"] == 16
    assert summary["evaluation_count"] == 2
    detail = client.get(f"/api/runs/{slow['run_id']}").json()
    assert detail == slow
    assert client.get(f"/api/runs/{slow['run_id']}/integrity").json()["valid"]
    evaluation = client.get(f"/api/analytics/evaluations?run_id={fast['run_id']}").json()
    assert evaluation["rows"][0]["results"]["quality"] == fast["summary"]["final_quality"]
    assert client.get("/api/analytics/resources?limit=1").json()["truncated"]


def test_live_run_is_visible_and_no_arbitrary_sql_or_files(tmp_path):
    client = TestClient(create_app(tmp_path))
    with RunRecorder(tmp_path, "live-dashboard", {}, telemetry_interval_seconds=0) as run:
        run.metric("distance", 4)
        assert client.get("/api/runs?status=running").json()[0]["run_id"] == run.run_id
        assert client.get("/api/metrics").json()[0]["value"] == 4
        assert not client.get(f"/api/runs/{run.run_id}/integrity").json()["valid"]
    assert client.get("/api/runs/invalid").status_code == 404
    assert client.get("/api/metrics?run_id=invalid").status_code == 422
    assert client.get("/api/analytics/secrets").status_code == 404
    assert client.post("/api/runs", json={"delete": True}).status_code == 405


def _run_dashboard_checks(checks):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the dashboard JavaScript contract check")
    script = Path(__file__).parents[1] / "src/gradientclimb/dashboard/dashboard.js"
    # Execute the shipped renderer with a minimal DOM. Network promises stay
    # pending: this test provides its complete canonical-record fixture directly.
    program = r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const document = {querySelector: () => ({value:'',addEventListener:()=>{},classList:{add:()=>{}}}),querySelectorAll:()=>[]};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8')+'\n'+process.argv[2],
 {document,fetch:()=>new Promise(()=>{}),assert});
"""
    result = subprocess.run(
        [node, "-e", program, str(script), checks],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_evaluation_without_optional_domain_tag_is_displayed():
    _run_dashboard_checks(r"""
const native = {run_id:'native-baseline',experiment_id:'test',algorithm:'always_gas',seed:1,
 status:'completed',environment:'actual_hill_climb_racing',metadata:{},duration:120,
 start_time:'2026-09-11T19:22:00Z'};
const synthetic = {...native,run_id:'synthetic',environment:'synthetic-quadratic'};
const surrogate = {...native,run_id:'surrogate',environment:'uncalibrated_hill_surrogate'};
const fixture = {runs:[native,synthetic,surrogate],metrics:[],resources:[],controls:[],
 evaluations:[{run_id:native.run_id,timestamp:'2026-09-11',
 protocol:'actual-screen-frozen-episode-1',results:{mean_distance:434.5,scope:'scripted baseline evaluation'}}]};
state.runs=fixture.runs;
assert.equal(domain(native),'real_game');
assert.notEqual(domain(synthetic),'real_game');
assert.notEqual(domain(surrogate),'real_game');
assert.equal(domain({...native,metadata:{evidence_domain:'explicit-domain'}}),'explicit-domain');
const html=overview(fixture);
assert.match(html,/Real-game evaluation[\s\S]*1 evaluations/);
assert.match(html,/scripted baseline evaluation/);
assert.match(html,/Simulator scores do not establish real-game competence/);
assert.match(benchmark(fixture),/One-hour outcome not measured/);
assert.match(overview({...fixture,evaluations:[]}),/Real-game evaluation[\s\S]*Not measured/);
""")


def test_checkpoint_targets_and_training_lineage_drive_benchmark_and_adaptation():
    _run_dashboard_checks(r"""
const targets=[5,10,20,30,45,60];
const checkpoint=(sha,target)=>({kind:'checkpoint',sha256:sha,metadata:{target_seconds:target*60}});
const cold={run_id:'cold-training',algorithm:'ppo',seed:42,status:'completed',
 configuration:{seconds:3600,profile:'default',terrain:'train'},metadata:{benchmark_class:'cold_start'},
 artifact_manifest:targets.map(t=>checkpoint('cold-'+t,t))};
const extended={...cold,run_id:'extended-training',parent_run:cold.run_id,parent_checkpoint:'cold-60',
 configuration:{seconds:600,profile:'default',terrain:'train'},metadata:{benchmark_class:'fine_tuning'},
 artifact_manifest:[checkpoint('extension-5',5)]};
const adapted={...extended,run_id:'adapted-training',configuration:{seconds:600,profile:'heavy',terrain:'rough'},
 artifact_manifest:[checkpoint('adapted-5',5),checkpoint('adapted-10',10),checkpoint('adapted-final',10)]};
const evaluationRun=(id,source)=>({run_id:id,parent_run:source.run_id,algorithm:'ppo',seed:42,
 status:'completed',vehicle_profile:'default',map_profile:'train',metadata:source.metadata});
const coldEval=evaluationRun('cold-evaluation',cold), extendEval=evaluationRun('extension-evaluation',extended),
 adaptEval=evaluationRun('adaptation-evaluation',adapted);
// The historical generalization record has no parent_run; the checkpoint hash resolves its owner.
const gridEval={...evaluationRun('generalization-evaluation',adapted),parent_run:null};
state.runs=[cold,extended,adapted,coldEval,extendEval,adaptEval,gridEval];
const evaluated=(run,sha,target,actual,profile='default',terrain='train')=>({run_id:run.run_id,
 protocol:'one-hour-v1',checkpoint_hash:sha,episodes:20,
 metadata:{requested_minutes:target,training_minutes:actual/60,training_elapsed_seconds:actual},
 results:{mean_distance:400,profile,terrain}});
const coldRows=targets.map(t=>evaluated(coldEval,'cold-'+t,t,t*60+1.27));
const extension=evaluated(extendEval,'extension-5',5,301.19);
const adaptation=[evaluated(adaptEval,'adapted-5',5,300.004,'heavy','rough'),
 evaluated(adaptEval,'adapted-10',10,600.084,'heavy','rough')];
const grid={run_id:gridEval.run_id,protocol:'held-out-surrogate-test-0.1',checkpoint_hash:'adapted-final',
 metadata:{condition:'new_vehicle_and_map'},results:{mean_distance:405,profile:'heavy',terrain:'rough',
 policy_config:{profile:'heavy',terrain:'rough'}}};
const data={runs:state.runs,evaluations:[...coldRows,extension,...adaptation,grid],metrics:[],resources:[],controls:[]};
assert.equal(checkpointTargetMinutes(coldRows[0]),5);
assert.equal(checkpointActualSeconds(coldRows[0]),301.27);
assert.equal(coldStartHour(coldRows[0]),true);
assert.equal(coldStartHour(extension),false);
const html=benchmark(data), governed=html.split('Continued or other checkpoint evaluations')[0];
assert.equal((governed.match(/class="checkpoint measured"/g)||[]).length,6);
assert.match(governed,/cold-evaluation/);
assert.doesNotMatch(governed,/extension-evaluation|adaptation-evaluation/);
assert.match(governed,/301.27 s/);
assert.doesNotMatch(governed,/5.021166666/);
assert.match(html,/extension-evaluation/);
assert.equal(adaptationMinutes(extension),null);
assert.equal(adaptationMinutes(adaptation[1]),600.084/60);
assert.equal(trainingRun(grid),adapted);
assert.equal(conditionName(trainedCondition(grid)),'heavy / rough');
assert.equal(evaluationContext(grid),'Adapted condition');
assert.equal(evaluationContext({...grid,results:{...grid.results,profile:'default',terrain:'train'}}),'Original-condition retention');
const transferred=transfer(data);
assert.match(transferred,/new_vehicle_and_map/);
assert.match(transferred,/Adapted condition/);
assert.match(transferred,/heavy \/ rough/);
assert.match(transferred,/Additional adaptation minutes/);
assert.match(transferred,/<title>ppo · seed 42 · adapted- · heavy \/ rough/);
assert.doesNotMatch(transferred,/<title>ppo · seed 42 · extended/);
// Neither an absent requested target nor an incomplete run can fill a governed badge.
const unlinked={...coldRows[0],checkpoint_hash:'unregistered',metadata:{training_minutes:5.021}};
assert.equal(checkpointTargetMinutes(unlinked),null);
assert.equal(coldStartHour(unlinked),false);
coldEval.status='failed';
assert.equal(coldStartHour(coldRows[0]),false);
""")
