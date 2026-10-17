"""SFTP upload to Almesbar (password auth via KU_ZSJ, same as almesbar_get.py)."""
import os
import sys
import time

import paramiko

HOST = 'login.almesbar.ku.ac.ae'
USER = '100065732@kunet.ae'


def env_lookup(name):
    value = os.environ.get(name)
    if value:
        return value.strip()
    try:
        import winreg
    except ImportError:
        return None
    for hive, key in ((winreg.HKEY_CURRENT_USER, 'Environment'),
                      (winreg.HKEY_LOCAL_MACHINE,
                       r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment')):
        try:
            with winreg.OpenKey(hive, key) as h:
                raw, _ = winreg.QueryValueEx(h, name)
        except OSError:
            continue
        if raw and str(raw).strip():
            return str(raw).strip()
    return None


def password():
    value = env_lookup('KU_ZSJ')
    if not value:
        raise SystemExit('KU_ZSJ is not set; cannot authenticate to Almesbar')
    if os.path.isfile(value):
        with open(value, 'rb') as fh:
            return fh.read().decode('utf-8', 'replace').lstrip('\ufeff').rstrip('\r\n')
    return value


def main():
    local, remote = sys.argv[1], sys.argv[2]
    cli = paramiko.SSHClient()
    cli.load_system_host_keys()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, username=USER, password=password(), look_for_keys=False,
                allow_agent=False, timeout=30, auth_timeout=60)
    try:
        sftp = cli.open_sftp()
        d = os.path.dirname(remote)
        if d:
            try:
                sftp.stat(d)
            except IOError:
                sftp.mkdir(d)
        t0 = time.time()
        sftp.put(local, remote)
        sftp.close()
        print('UPLOADED %s -> %s (%.0f KB in %.1f s)' % (
            local, remote, os.path.getsize(local) / 1024.0, time.time() - t0))
    finally:
        cli.close()


if __name__ == '__main__':
    main()
