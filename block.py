"""MEGA-ASR integration uses only the public block execution and storage contract."""

import json
from bloxsmith_app.block_api import BlockDefinition, BlockRuntimeOutput, BlockRuntimePreparation, BlockRuntimeResult
from . import logic, ui


# FB1 - Accept completed local files/recording events without replaying another port.
# FB2 - Verify pinned engine/model files, load actual offline inference in a private process.
# FB3 - Transcribe bounded chunks and publish only a complete validated final result.
# FB4 - Bound time/resources, cancel owned descendants and preserve original recordings.
# FB5 - Support both public runtimes/all package origins and responsive EN/FR draft forms.
class MegaAsrBlock(BlockDefinition):
    kind = "mega_asr"

    def ui_assets(self, surface="modal"):
        return list(self.model.get("ui_assets", {}).get(surface, []))

    def prepare_runtime(self, context):
        logic.configuration(context.config)
        return BlockRuntimePreparation()

    def execute_runtime(self, context):
        try:
            config = logic.configuration(context.config)
            if context.input_events:
                names = {int(port.id): port.name for port in context.input_ports}
                values = [event.value for event in context.input_events if names.get(event.input_port_id) == "audio"]
                if len(values) != len(context.input_events):
                    raise logic.MegaError("Unexpected input port.")
            else:
                values = [context.input_value("audio", "1")] if context.has_input_value("audio", "1") else []
            if not values:
                return BlockRuntimeResult(status="skipped", outputs=[])
            if len(values) != 1:
                raise logic.MegaError("Send one completed audio recording per activation.")
            storage = context.services.get("get_block_storage_dir")
            if not callable(storage):
                raise logic.MegaError("The block storage service is unavailable.")
            result = logic.transcribe(values[0], config, context.root_dir, storage(),
                context.services.get("cancel_requested", lambda: False))
            values = {"metadata": result, "status": {"code": "transcribed" if result["text"].strip() else "no_text",
                "duration_sec": result["duration_sec"], "elapsed_sec": result["elapsed_sec"], "source": result["source"]}}
            if result["text"].strip():
                values["transcript"] = result["text"]
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            detail = str(exc) if isinstance(exc, logic.MegaError) else "Local transcription is unavailable; check the completed file and prerequisites."
            values = {"status": {"code": "cancelled" if isinstance(exc, logic.Cancelled) else "error", "detail": detail}}
        outputs = [BlockRuntimeOutput(port_id=int(port.id), port_name=port.name,
            value=values[port.name] if port.name == "transcript" else json.dumps(values[port.name], ensure_ascii=True, allow_nan=False),
            content_type="text/plain" if port.name == "transcript" else "application/json")
            for port in context.output_ports if port.name in values]
        return BlockRuntimeResult(status="success", outputs=outputs, metadata={self.kind: values["status"]},
            logs=["[mega-asr] " + values["status"]["code"]])

    def handle_ui_action(self, *, node, action, values, payload=None):
        if action in {"save_settings", "modal_update_fields", "inspector_update_fields"}:
            patch = (values or {}).get("node_patch") or {}
            if "config" in patch:
                try:
                    logic.configuration({**self.default_config(), **(node.get("config") or {}), **patch["config"]})
                except (ValueError, TypeError) as exc:
                    return {"error": self.translate("block.mega_asr.error", {"detail": str(exc)}, fallback="Transcription failed: {detail}")}
        return ui.replace_config(super().handle_ui_action(node=node,
            action="modal_update_fields" if action == "save_settings" else action, values=values, payload=payload), node)

    def render_modal(self, *, node, payload=None):
        return ui.modal(self, node, payload)

    def render_inspector_panel(self, *, node, payload=None):
        return ui.inspector(self, node, payload)

    def render_node_card(self, *, node, payload=None):
        config = {**self.default_config(), **(node.get("config") or {})}
        language = self.translate("block.mega_asr.option_" + config["language"], fallback=config["language"])
        return ui.card(self, node, language + " · " + config["device"] + " · " + config["routing"])
