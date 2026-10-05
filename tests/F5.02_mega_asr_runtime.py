"""FB1/FB2/FB3/FB4/FB5: real model, both runtimes, all package origins and Run Stop."""

import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(Path(__file__).parent)]
from blocs.mega_asr.block import MegaAsrBlock
from block_test_packages import install_test_package, prepare_release_run, surface_payload
from mega_fixture import settings, speech, worker_pids
from ui_smoke_common import (isolated_server, http_json, graph_payload, text_node, data_edge,
    create_project_api, graph_storage_dir, create_run_api, wait_for_run_terminal,
    wait_for_run_predicate, stop_run_api)


def value(data, port):
    text = data.get("output_values", {}).get("asr:"+str(port), {}).get("value")
    return json.loads(text) if text and port != 1 else text or ({} if port != 1 else "")


def main():
    for mode in ("centralized", "zeromq_active"):
        for origin in (None, "managed", "linked"):
            with isolated_server() as server:
                audio = speech(server.root_dir)
                node = MegaAsrBlock().build_node_payload(node_id="asr", position={"x":430,"y":160},
                    config_overrides=settings(server.root_dir))
                node["outputs"].reverse()
                if origin:
                    model = install_test_package(server,"mega_asr",origin=origin);node["block_version"] = model["version"]
                    for surface in ("modal","inspector_panel","node_card"):
                        assert "{{" not in surface_payload(server,model,node,surface)["html"]
                graph = graph_payload("Offline MEGA ASR",[text_node("source","Completed recording","",60,160),node],
                    [data_edge("audio","source",1,"asr",1)])
                project = create_project_api(server,document=graph)["project"]
                storage = graph_storage_dir(server,project)/"instances"/"1"/"blocs"/"asr"
                if mode == "centralized":
                    def send(path):
                        graph["nodes"][0]["outputs"][0]["text"] = str(path)
                        (graph_storage_dir(server,project)/"graph.json").write_text(json.dumps({**graph,"graph_id":project["graph_id"]}),encoding="utf-8")
                        http_json(server.base_url,f"/api/projects/{project['graph_id']}/graph/reload",method="POST",payload={})
                        run = create_run_api(server,graph,project_id=project["graph_id"],runtime_mode=mode)
                        data = wait_for_run_terminal(server,run["run_id"],timeout_sec=190)
                        assert data["status"]=="success",data.get("logs")
                        return data
                    data = send(audio)
                    assert value(data,3)["code"]=="transcribed",data.get("logs")
                    assert "weather" in value(data,1).lower() and value(data,2)["final"]
                    assert all(c["route_source"]=="router" for c in value(data,2)["chunks"])
                    data = send(server.root_dir/"missing.wav")
                    assert value(data,3)["code"]=="error" and not value(data,1)
                    assert not list(storage.glob("mega-asr-job-*"))
                else:
                    rid = prepare_release_run(server,project["graph_id"],graph)["run_id"]
                    assert not list(storage.glob("mega-asr-job-*")),"Preparation loaded the model"
                    http_json(server.base_url,f"/api/runs/{rid}/play",method="POST",payload={})
                    def send(path):
                        http_json(server.base_url,f"/api/runs/{rid}/active/control",method="POST",payload={"action":"publish_output",
                            "node_id":"source","port_id":1,"value":str(path),"content_type":"file/path"})
                    def wait(predicate):
                        data = wait_for_run_predicate(server,rid,lambda item:item.get("status")=="failed" or predicate(item),
                            "MEGA ASR did not finish",timeout_sec=190)
                        assert data["status"]!="failed",data.get("logs")
                        return data
                    stopped = False
                    try:
                        assert not list(storage.glob("mega-asr-job-*")),"Play without input loaded the model"
                        send(audio)
                        # Reversed ports publish status first. Wait for the complete
                        # asynchronous output set, not merely the first received event.
                        data = wait(lambda d:value(d,3).get("code")=="error" or (
                            value(d,3).get("code")=="transcribed" and bool(value(d,1)) and value(d,2).get("final")))
                        assert value(data,3)["code"]=="transcribed",data.get("logs")
                        assert "weather" in value(data,1).lower() and value(data,2)["final"],data.get("output_values")
                        send(server.root_dir/"missing.wav")
                        wait(lambda d:value(d,3).get("code")=="error")
                        assert not list(storage.glob("mega-asr-job-*"))
                        # Stop a new actual loading job, not a sleep pretending to be ASR.
                        send(audio)
                        deadline = time.monotonic()+15
                        jobs, native_pids = [], []
                        while time.monotonic()<deadline:
                            jobs = list(storage.glob("mega-asr-job-*/input.wav"))
                            job_dirs = {str(job.parent) for job in jobs}
                            native_pids = worker_pids(job_dirs)
                            if native_pids:
                                break
                            time.sleep(.02)
                        assert native_pids,"Second native inference did not start"
                        started = time.monotonic();stop_run_api(server,rid)
                        ended = wait_for_run_terminal(server,rid,timeout_sec=15);stopped=True
                        assert time.monotonic()-started<8 and "shutdown_timeout" not in str(ended.get("logs")),ended.get("logs")
                        # These PIDs were observed inside the actual job before Stop.
                        from mega_fixture import alive
                        deadline = time.monotonic()+3
                        while any(alive(pid) for pid in native_pids) and time.monotonic()<deadline:
                            time.sleep(.02)
                        assert not any(alive(pid) for pid in native_pids),"Native inference survived Run Stop"
                    finally:
                        if not stopped:
                            stop_run_api(server,rid);wait_for_run_terminal(server,rid,timeout_sec=15)
            print("[ok] MEGA actual CPU runtime + offline assets + Stop "+mode+" "+str(origin),flush=True)


if __name__=="__main__":
    main()
