# Host smoke test

Run in a real Omarchy desktop session with NVIDIA driver access. `doctor` should show the tested model and a Stable MM implementation. Verify the top bar icon opens, GPU memory refreshes, and the Model Path control saves a path if needed. Use an existing small JPEG or PNG for the automated short image check:

```bash
./scripts/install.sh
local-exl3 doctor
python scripts/smoke.py /absolute/path/to/small-test-image.jpg
```

The script starts each profile, checks `/v1/models`, sends one short text request and one short image request, then stops. It does not perform a context benchmark. Check `local-exl3 logs` for OOM/CUDA errors and use **Open Pi** from the panel after starting either profile.

Manual equivalent for text and status:

```bash
./scripts/install.sh
local-exl3 doctor
local-exl3 start --profile balanced
local-exl3 status --json
curl -fsS http://127.0.0.1:8881/v1/models
curl -fsS http://127.0.0.1:8881/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"exllama-v3","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8}'
local-exl3 stop
local-exl3 start --profile max-context
local-exl3 status --json
curl -fsS http://127.0.0.1:8881/v1/models
curl -fsS http://127.0.0.1:8881/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"exllama-v3","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8}'
local-exl3 stop
```

The image request is handled by `scripts/smoke.py`. Check the reply and server log for OOM/CUDA errors. Do not run long context benchmarks as part of this smoke test.
