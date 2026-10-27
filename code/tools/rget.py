"""SFTP download from <CLUSTER>."""
import os, sys, time, paramiko

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
    remote, local = sys.argv[1], sys.argv[2]
    cli = paramiko.SSHClient()
    cli.load_system_host_keys()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, username=USER, pkey=load_key(), look_for_keys=False,
                allow_agent=False, timeout=30, auth_timeout=60)
    try:
        sftp = cli.open_sftp()
        d = os.path.dirname(local)
        if d:
            os.makedirs(d, exist_ok=True)
        t0 = time.time()
        sftp.get(remote, local)
        sftp.close()
        print('DOWNLOADED %s -> %s (%.0f KB in %.1f s)' % (
            remote, local, os.path.getsize(local) / 1024.0, time.time() - t0))
    finally:
        cli.close()

if __name__ == '__main__':
    main()
