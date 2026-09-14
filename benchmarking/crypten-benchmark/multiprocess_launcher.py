#!/usr/bin/env python3
# Adapted from CrypTen examples/multiprocess_launcher.py (MIT license).
# https://github.com/facebookresearch/CrypTen

from __future__ import annotations

import logging
import multiprocessing
import os
import uuid
from typing import Any, Callable, Optional

import crypten


class MultiProcessLauncher:
    def __init__(
        self,
        world_size: int,
        run_process_fn: Callable[..., None],
        fn_args: Optional[Any] = None,
    ) -> None:
        env = os.environ.copy()
        env["WORLD_SIZE"] = str(world_size)
        multiprocessing.set_start_method("spawn", force=True)

        init_method = f"file:///tmp/crypten-rendezvous-{uuid.uuid1()}"
        env["RENDEZVOUS"] = init_method

        self.processes = []
        for rank in range(world_size):
            process = multiprocessing.Process(
                target=self._run_process,
                name=f"crypten-rank-{rank}",
                args=(rank, world_size, env, run_process_fn, fn_args),
            )
            self.processes.append(process)

        if crypten.mpc.ttp_required():
            ttp_process = multiprocessing.Process(
                target=self._run_process,
                name="TTP",
                args=(world_size, world_size, env, crypten.mpc.provider.TTPServer, None),
            )
            self.processes.append(ttp_process)

    @classmethod
    def _run_process(
        cls,
        rank: int,
        world_size: int,
        env: dict[str, str],
        run_process_fn: Callable[..., None],
        fn_args: Optional[Any],
    ) -> None:
        for key, value in env.items():
            os.environ[key] = value
        os.environ["RANK"] = str(rank)
        logging.getLogger().setLevel(logging.INFO if rank == 0 else logging.WARNING)
        crypten.init()
        if fn_args is None:
            run_process_fn()
        else:
            run_process_fn(fn_args)

    def start(self) -> None:
        for process in self.processes:
            process.start()

    def join(self) -> None:
        for process in self.processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"{process.name} exited with code {process.exitcode}"
                )

    def terminate(self) -> None:
        for process in self.processes:
            process.terminate()
