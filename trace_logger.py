import json
import os
import time
import uuid
from typing import Dict, Any, Optional, List

TRACES_FILE = os.path.join(os.path.dirname(__file__), "traces.jsonl")


class TraceLogger:
    """
    OpenJarvis Trace Collection Primitive.
    Logs interaction traces (speech input, LLM reasoning, tool calls, and results)
    to build the continuous local fine-tuning dataset for Jarvis.
    """
    def __init__(self, log_path: str = TRACES_FILE):
        self.log_path = log_path

    def log_interaction(
        self,
        query: str,
        response: str,
        tool_called: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        tool_output: Optional[str] = None,
        latency_ms: float = 0.0,
        model_name: str = "jarvis",
        verified: bool = True,
        success: bool = True,
        feedback: Optional[str] = None
    ) -> str:
        """
        Appends an interaction trace to traces.jsonl.
        Returns the unique trace_id.
        """
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        record = {
            "trace_id": trace_id,
            "timestamp": time.time(),
            "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verified_owner": verified,
            "model": model_name,
            "user_query": query,
            "tool_called": tool_called,
            "tool_args": tool_args or {},
            "tool_output": tool_output,
            "assistant_response": response,
            "latency_ms": round(latency_ms, 2),
            "success": success,
            "feedback": feedback
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[TraceLogger] Failed to write trace ({e})")

        return trace_id

    def load_traces(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Loads traces from disk."""
        if not os.path.exists(self.log_path):
            return []
        traces = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        traces.append(json.loads(line))
        except Exception as e:
            print(f"[TraceLogger] Error reading traces: {e}")
        if limit:
            return traces[-limit:]
        return traces

    def export_to_sft_dataset(self, output_path: str = "dataset_jarvis_sft.json") -> int:
        """
        Transforms collected traces into a standardized instruction fine-tuning dataset
        (Alpaca / ChatML format) ready for QLoRA training.
        """
        traces = self.load_traces()
        dataset = []

        system_instruction = (
            "You are J.A.R.V.I.S., a witty, hyper-competent British desktop AI assistant. "
            "Execute tools when required and reply in concise, spoken sentences."
        )

        for t in traces:
            # Only include successful and non-empty traces
            if not t.get("success", True) or not t.get("user_query"):
                continue

            query = t["user_query"]
            tool = t.get("tool_called")
            resp = t.get("assistant_response", "")

            if tool:
                # Format with tool call representation
                target_output = f"<tool_call>\n{{\"name\": \"{tool}\", \"arguments\": {json.dumps(t.get('tool_args', {}))}}}\n</tool_call>"
            else:
                target_output = resp

            dataset.append({
                "instruction": system_instruction,
                "input": query,
                "output": target_output
            })

        out_abs = os.path.join(os.path.dirname(__file__), output_path)
        with open(out_abs, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"[TraceLogger] Exported {len(dataset)} training examples to {out_abs}")
        return len(dataset)
