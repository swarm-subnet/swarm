#!/usr/bin/env python3
import sys
import os

if len(sys.argv) > 1 and sys.argv[1] == "container_launcher":
    from swarm.core.container_launcher import main
    sys.argv = sys.argv[1:]
    main()
elif len(sys.argv) > 1 and sys.argv[1] == "VERIFY_ONLY":
    from swarm.core.evaluator import main
    sys.argv = sys.argv[1:]
    main()
else:
    from swarm.core.evaluator import main
    main()

