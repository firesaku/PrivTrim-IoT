"""SFTP upload to <CLUSTER> (shared /dpc), with progress.

Usage: python rput.py <local> <remote>
"""
import os
import sys
import time

import paramiko

HOST = 'login.<CLUSTER>.<ADMIN_DOMAIN>'
USER = '100065732_ku'
KEY = r'D:\Shijie Zeng\HPC\100065732_ku-ssh-key.pem'
PWFILE = r'D:\Shijie Zeng\HPC\<CLUSTER>-key-pass.txt'


def load_key():
    with open(PWFILE, 'rb') as fh:
        pw = fh.read().decode('utf-8', 'replace').lstrip('\ufeff').rstrip('\r\n')
    for cls in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
        try:
            return cls.from_private_key_file(KEY, password=pw)
        except Exception:
            continue
    raise SystemExit('cannot load key')


def main():
    local, remote = sys.argv[1], sys.argv[2]
    size = os.path.getsize(local)
    cli = paramiko.SSHClient()
    cli.load_system_host_keys()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, username=USER, pkey=load_key(), look_for_keys=False, allow_agent=False,
                timeout=30, auth_timeout=60)
    try:
        sftp = cli.open_sftp()
        t0 = time.time()
        sent = [0]

        def cb(done, total):
            sent[0] = done
            if time.time() - t0 > 3 and int(time.time() - t0) % 5 == 0:
                print('   %.0f/%.0f MB  %.1f MB/s' % (done / 1e6, total / 1e6,
                                                      done / 1e6 / max(time.time() - t0, 1e-6)), flush=True)

        sftp.put(local, remote, callback=cb)
        sftp.close()
        st = cli.exec_command('ls -l %s; sha256sum %s' % (remote, remote))[1].read().decode()
        print('UPLOADED %s -> %s (%.0f MB in %.1f s)' % (local, remote, size / 1e6, time.time() - t0))
        print(st.strip())
    finally:
        cli.close()


if __name__ == '__main__':
    main()
