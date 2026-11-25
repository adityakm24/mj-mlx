#!/bin/zsh
# stop | start the login services that load the machine during benchmarks
L=(local.service-a local.service-b sh.brew.openshell local.dev-backend)
for l in $L; do
  if [[ $1 == stop ]]; then launchctl bootout gui/501/$l 2>/dev/null && echo "stopped $l"
  else launchctl print gui/501/$l >/dev/null 2>&1 || { launchctl bootstrap gui/501 ~/Library/LaunchAgents/$l.plist && echo "started $l"; }; fi
done
