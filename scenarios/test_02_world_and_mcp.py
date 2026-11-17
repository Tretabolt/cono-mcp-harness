import json
import subprocess
import sys
import unittest
from pathlib import Path
import scenarios  # noqa: F401


class TestMCPSubprocessBoundary(unittest.TestCase):
    def test_mcp_stdio_jail_and_affordance(self) -> None:
        root = Path(__file__).resolve().parent.parent
        server_path = root / "packages" / "mcp_server" / "server.py"
        env = {"PYTHONPATH": str(root / "packages")}

        proc = subprocess.Popen(
            [sys.executable, str(server_path), "1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        assert proc.stdin and proc.stdout

        def rpc(msg: dict) -> dict:
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            return json.loads(line)

        try:
            init = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
            self.assertEqual(init["result"]["serverInfo"]["name"], "cono-mcp-harness")

            obs = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "inspect"}})["result"]["content"][0]["data"]
            self.assertFalse(obs["contorno_ciano"])
            self.assertNotIn("walls", obs)  # Estado oculto S nunca vaza!

            # No nivel 1: agente comeca em (2,2) yaw=90 e Cubo esta em (4,2). 1 passo -> (3,2) -> contorno_ciano True!
            obs2 = rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "move", "arguments": {"direction": "frente"}}})["result"]["content"][0]["data"]
            self.assertTrue(obs2["contorno_ciano"])
            self.assertEqual(obs2["front_label"], "Cubo")

            # Tentativa de ferramenta fora do contrato deve ser bloqueada pela AI-Jail
            err = rpc({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "read_host_disk"}})
            self.assertIn("error", err)
        finally:
            proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
