import json
import subprocess
import os
import time

class ShowdownClient:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        bridge_path = os.path.join(self.engine_path, 'bridge.js')
        
        # Start the node process
        self.process = subprocess.Popen(
            ['node', bridge_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.engine_path
        )
        
        # Wait for the ready signal
        self._read_response()
        
    def _send_request(self, req_dict):
        self.process.stdin.write(json.dumps(req_dict) + '\n')
        self.process.stdin.flush()
        
    def _read_response(self):
        line = self.process.stdout.readline()
        if not line:
            error_output = self.process.stderr.read()
            raise RuntimeError(f"Node bridge process terminated unexpectedly. Stderr: {error_output}")
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON from node bridge: {line}") from e

    def init_battle(self, formatid='gen3randombattle', p1_team=None, p2_team=None):
        req = {
            "type": "init",
            "formatid": formatid
        }
        if p1_team: req["p1_team"] = p1_team
        if p2_team: req["p2_team"] = p2_team
        
        self._send_request(req)
        resp = self._read_response()
        if resp.get('type') == 'error':
            raise Exception(f"Error initializing battle: {resp.get('error')}")
        return resp

    def get_result(self, state, p1_action, p2_action=None):
        req = {
            "type": "result",
            "state": state,
            "p1_action": p1_action
        }
        if p2_action: req["p2_action"] = p2_action
        
        self._send_request(req)
        resp = self._read_response()
        if resp.get('type') == 'error':
            raise Exception(f"Error executing action: {resp.get('error')}\nStack: {resp.get('stack')}")
        return resp

    def close(self):
        if self.process:
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait()

if __name__ == "__main__":
    # Test
    client = ShowdownClient(os.path.abspath("../battle_engine"))
    print("Bridge ready!")
    res = client.init_battle(formatid="gen3randombattle")
    print("Initial state received. Actions available:")
    request = res.get('request', {})
    if 'active' in request and request['active']:
        for move in request['active'][0]['moves']:
            print(f" - move {move['move']}")
            
    print("\nExecuting a move...")
    res2 = client.get_result(res['state'], p1_action="move 1")
    print("New state received!")
    client.close()
