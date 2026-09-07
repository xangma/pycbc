# Pass 2 run record

Host: len. Cwd: /home/xangma/pycbc-torch-profile-20260907-r3.
Command: /home/xangma/pycbc-torch-split-20260905/venv/bin/python profile-campaign.py --shared-host.
PID/PGID: 1591583. Started 2026-09-07 20:05:20 UTC.
Logs: launch.log, stages/*.log, stages/*.stderr; status: profile-status.json.
Stop: ssh len 'kill -TERM 1591583'. Each stage timeout: 900 seconds; owned process-group cleanup and inherited flock.
Initial Linux 9 control tests PASS with zero skips (including perf inheritance), runtime 2 tests PASS (18 backend/failure combinations), frozen comparator self-test PASS. Qualifications started. Next check within 60 seconds while active; share validated attribution before repeated timing completes.

A pre-staging checksum attempt during transfer rejected the incomplete archive and created no acquisition directory. After transfer completed, its exact hash passed and staging succeeded. Existing evidence was not changed.

All 49 science/control stages and 21 science workers completed; final state complete. All profiles, 3 qualifications, 9 timings and parity passed. Archive export PID/PGID 2357490 completed, then local scp PID 35856 completed. Final release audit at 20:38:55 UTC found all 52 science/export groups inactive and independently reacquired the lock. Explicit optimizer release sent after full evidence transfer and replay. No active job or next check remains. See REPORT.md and handoff.json.
