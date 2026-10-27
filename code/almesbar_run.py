"""
Almesbar 远程执行器（密码认证）。
凭据来源：环境变量 KU_ZSJ -> 指向一个只含密码的文件；密码只读进进程内存，
不写入本文件、不拼进命令行、不打印。仅用于本用户已获授权的 <SLURM_ACCOUNT> 项目。
"""
import io, os, sys, argparse
import paramiko

HOST = 'login.almesbar.<ADMIN_DOMAIN>'
USER = '100065732@kunet.ae'

def load_password():
    p = os.environ.get('KU_ZSJ')
    if not p:
        raise SystemExit('KU_ZSJ not set')
    if os.path.isfile(p):
        with io.open(p, 'rb') as fh:
            return fh.read().decode('utf-8', 'replace').lstrip('\ufeff').rstrip('\r\n')
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--command', default=None)
    ap.add_argument('--script', default=None)
    ap.add_argument('--timeout', type=int, default=180)
    a = ap.parse_args()
    if not a.command and not a.script:
        raise SystemExit('need --command or --script')

    pw = load_password()
    cli = paramiko.SSHClient()
    cli.load_system_host_keys()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, username=USER, password=pw, look_for_keys=False,
                allow_agent=False, timeout=30, auth_timeout=60)
    pw = None
    try:
        cmd = a.command
        if a.script:
            with io.open(a.script, encoding='utf-8') as fh:
                cmd = 'bash -s'
                stdin, stdout, stderr = cli.exec_command(cmd, timeout=a.timeout)
                stdin.write(fh.read()); stdin.channel.shutdown_write()
            out = stdout.read().decode('utf-8', 'replace')
            err = stderr.read().decode('utf-8', 'replace')
            rc = stdout.channel.recv_exit_status()
        else:
            stdin, stdout, stderr = cli.exec_command(cmd, timeout=a.timeout)
            out = stdout.read().decode('utf-8', 'replace')
            err = stderr.read().decode('utf-8', 'replace')
            rc = stdout.channel.recv_exit_status()
        sys.stdout.write(out)
        if err.strip():
            sys.stdout.write('\n[stderr]\n' + err)
        sys.stdout.write('\n[exit %d]\n' % rc)
    finally:
        cli.close()

if __name__ == '__main__':
    main()
