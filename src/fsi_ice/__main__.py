"""以 ``python -m fsi_ice`` 方式运行时的模块入口。

把命令行参数交给 :func:`fsi_ice.cli.main`，并用其返回的退出码退出进程，
使子命令的退出码语义与直接调用 ``fsi-ice`` 脚本保持一致。
"""

from fsi_ice.cli import main

raise SystemExit(main())
