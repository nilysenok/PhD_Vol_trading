#!/usr/bin/env python3
"""Хуки канона v23 (§0 «три слоя принуждения»). Только stdlib, без сети.

Вызов: sdf_hook.py pre | post | stop | precompact | start   (JSON события — на stdin).
Отказ PreToolUse / замечание PostToolUse / запрет остановки — код 2 и сообщение в stderr.
Списки путей и команд — zamki.json рядом (формат "v": 1).
"""
import json
import os
import re
import sys
import time
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, 'zamki.json'), encoding='utf-8') as _f:
    K = json.load(_f)
MAIN = os.path.realpath(K['osnovnaya_kopiya'])
AUD = os.path.join(MAIN, '.claude', 'audit')
HANDOFF = os.path.join(MAIN, '.claude', 'handoff')
RX_ZAKR = [re.compile(x) for x in K['zakryto_chtenie']]
RX_ZAM = [re.compile(x) for x in K['zamorozheno']]
PREF = '[хук v23] '


# ---------- пути ----------

def koren(p):
    """Корень git-копии (папка с .git), содержащей путь p; None — вне копий."""
    d = p if os.path.isdir(p) else os.path.dirname(p)
    while d and d != '/':
        if os.path.exists(os.path.join(d, '.git')):
            return d
        d = os.path.dirname(d)
    return None


def norm(p, cwd):
    p = os.path.expanduser(p)
    if not os.path.isabs(p):
        p = os.path.join(cwd, p)
    return os.path.realpath(p)


def rel(p):
    r = koren(p)
    return os.path.relpath(p, r) if r else p


def pod(p, d):
    return p == d or p.startswith(d.rstrip('/') + '/')


def zakryto(p):
    r = rel(p)
    return any(x.search(r) or x.search(p) for x in RX_ZAKR)


def zamorozheno(p):
    r = rel(p)
    if os.path.basename(p) in K['konfig_testov']:
        return 'конфиг тестов'
    return 'замороженный файл' if any(x.search(r) for x in RX_ZAM) else None


def v_osnovnoy(p, own):
    """Запись в основную копию (или чужую рабочую) из своей рабочей копии."""
    return own != MAIN and pod(p, MAIN) and not pod(p, own)


def predok_zakrytogo(d, own):
    for r in {own, MAIN}:
        for z in K['zakrytye_papki']:
            if pod(os.path.join(r, z), d):
                return True
    return False


def zakrytyy_test(p):
    return rel(p) in K['zakrytye_testy']


def otkaz(chto, pravilo, kak=''):
    sys.stderr.write(PREF + 'ОТКАЗ: ' + chto + '\nПравило: ' + pravilo + ('\nКак законно: ' + kak if kak else '') + '\n')
    sys.exit(2)


R_DANNYE = '.claude/rules/dannye.md п. 2–4 (данные ≥ 2022, test_locked/, замки OPENED_*)'
R_ZAM = '.claude/rules/polnota.md п. 7 (замки не снимаются); dannye.md п. 4–5; pravki.md п. 11 (эталоны — только в архив)'
R_PRAVKI = 'CLAUDE.md «Запреты»: файлы правятся только инструментами правки (Edit/Write); задания: «без sed -i, heredoc, cat >»'
R_KOPIYA = 'orkestraciya.md абз. 2 (писать только в своём списке файлов); dev_log_ET2 27.10 11:02 — запись в основную копию'
R_NEOBR = 'CLAUDE.md «Когда останавливаться»: необратимое; задания: «без rm -rf», force-push запрещён правом'
R_TESTY = 'testy.md п. 7 и orkestraciya.md абз. 3 (автор не меняет тесты и их конфиг)'


# ---------- «да» владельца на открытие ----------

def sha_file(p, limit=200 * 2 ** 20):
    try:
        if os.path.getsize(p) > limit:
            return 'больше-200МБ'
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(2 ** 20), b''):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def da_ok(skript_abs):
    p = os.path.join(MAIN, K['da_fail'])
    try:
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
    except (OSError, ValueError):
        return 'нет файла «да» владельца ' + K['da_fail']
    if d.get('v') != 1 or d.get('da') != 'да':
        return 'в файле «да» нет "v": 1 и "da": "да"'
    if d.get('skript') != rel(skript_abs):
        return '«да» дано на %s, а запускается %s' % (d.get('skript'), rel(skript_abs))
    if d.get('data') != time.strftime('%d.%m.%Y'):
        return '«да» датировано %s, сегодня %s' % (d.get('data'), time.strftime('%d.%m.%Y'))
    if d.get('sha256') != sha_file(skript_abs):
        return 'SHA скрипта не совпадает с SHA в «да» (скрипт меняли после «да»)'
    return None


# ---------- разбор Bash ----------

HEREDOC = re.compile(r"(?<!<)<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
SEP = {';', '&&', '||', '|', '&', '|&', '(', ')', ';;'}
WRAP = {'sudo', 'env', 'nice', 'nohup', 'time', 'timeout', 'command', 'exec', 'caffeinate', 'xargs'}
PYRX = re.compile(r'^python(3(\.\d+)?)?$')
WRITE = {'cp', 'mv', 'rm', 'rmdir', 'touch', 'mkdir', 'ln', 'install', 'rsync', 'truncate', 'chmod', 'chown', 'unlink'}
DEST_ONLY = {'cp', 'install', 'rsync', 'ln'}
AVTOR = {'cat', 'echo', 'printf', 'sed', 'awk', 'perl', 'gsed', 'gawk'}
RO = {'ls', 'cat', 'head', 'tail', 'grep', 'egrep', 'rg', 'wc', 'find', 'du', 'stat', 'file', 'diff', 'cmp',
      'shasum', 'sha256sum', 'md5', 'md5sum', 'date', 'echo', 'printf', 'pwd', 'cd', 'test', '[', 'true', 'sort',
      'uniq', 'cut', 'tr', 'awk', 'sed', 'jq', 'less', 'tree', 'realpath', 'basename', 'dirname', 'readlink'}
GIT_RO = {'status', 'log', 'diff', 'show', 'ls-files', 'ls-tree', 'rev-parse', 'blame', 'grep', 'cat-file',
          'describe', 'merge-base', 'shortlog', 'fetch', 'remote', 'reflog', 'for-each-ref', 'name-rev', 'check-ignore'}
MSG_OPTS = {'-m', '--message', '--title', '-t', '--body', '-b', '-F', '--file', '--subject'}


def razbor(cmd):
    """-> (сегменты, есть_heredoc). Сегмент — список слов shlex."""
    import shlex
    lines, out, i, hd = cmd.split('\n'), [], 0, False
    while i < len(lines):
        out.append(lines[i])
        ms = list(HEREDOC.finditer(lines[i]))
        i += 1
        for m in ms:
            hd = True
            while i < len(lines) and lines[i].strip() != m.group(2):
                i += 1
            i += 1
    s = ' ; '.join(out)
    try:
        lx = shlex.shlex(s, posix=True, punctuation_chars=True)
        lx.whitespace_split = True
        lx.commenters = ''
        toks = list(lx)
    except ValueError:
        toks = s.split()
    segs, cur = [], []
    for t in toks:
        if t in SEP:
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        segs.append(cur)
    return segs, hd


def segment(seg):
    """-> env, слова, цели перенаправления >."""
    env, words, redir, i = {}, [], [], 0
    while i < len(seg):
        t = seg[i]
        if t and set(t) <= set('<>&|') and '>' in t:
            if i + 1 < len(seg):
                tg = seg[i + 1]
                if not (t.endswith('&') and tg.isdigit()):
                    redir.append(tg)
            i += 2
            continue
        if t and set(t) <= set('<>&-') and '<' in t:
            i += 2
            continue
        if t.isdigit() and i + 1 < len(seg) and set(seg[i + 1]) <= set('<>&|'):
            i += 1
            continue
        if not words and re.match(r'^[A-Za-z_][A-Za-z0-9_]*=', t):
            k, v = t.split('=', 1)
            env[k] = v
            i += 1
            continue
        words.append(t)
        i += 1
    while words and os.path.basename(words[0]) in WRAP:
        w = os.path.basename(words.pop(0))
        if w == 'nice' and words[:1] == ['-n']:
            words = words[2:]
        elif w in ('nice', 'timeout') and words and re.match(r'^-?[\d.]+[smhd]?$', words[0]):
            words.pop(0)
        elif w == 'env':
            while words and '=' in words[0] and not words[0].startswith('-'):
                k, v = words.pop(0).split('=', 1)
                env[k] = v
    return env, words, redir


def kandidaty(t):
    """Строки-кандидаты в пути из одного слова."""
    out = [t]
    if '=' in t:
        out.append(t.split('=', 1)[1])
    return [x for x in out if x and not x.startswith('-')]


def raskryt(x, cwd):
    if any(c in x for c in '*?['):
        import glob
        pat = x if os.path.isabs(x) else os.path.join(cwd, x)
        res = []
        for n, g in enumerate(glob.iglob(os.path.expanduser(pat), recursive=True)):
            if n > 5000:
                break
            res.append(os.path.realpath(g))
        return res + [norm(x, cwd)]
    return [norm(x, cwd)]


def py_cel(words):
    """-> ('m', модуль) | ('s', путь) | ('c', '') | None, аргументы после цели."""
    if words and words[0].endswith('.py'):
        return ('s', words[0]), words[1:]
    i = 1
    while i < len(words):
        w = words[i]
        if w == '-m':
            return ('m', words[i + 1] if i + 1 < len(words) else ''), words[i + 2:]
        if w == '-c':
            return ('c', ''), words[i + 1:]
        if w.startswith('-'):
            i += 1
            continue
        return ('s', w), words[i + 1:]
    return None, []


def pytest_args(words):
    if os.path.basename(words[0]) in ('pytest', 'py.test'):
        return words[1:]
    if PYRX.match(os.path.basename(words[0])):
        c, args = py_cel(words)
        if c and c[0] == 'm' and c[1] == 'pytest':
            return args
    return None


PT_ARG = {'-k', '-m', '-p', '-o', '-c', '-n', '-W', '--tb', '-r', '--rootdir', '--maxfail', '--durations',
          '--deselect', '--ignore', '--confcutdir', '--basetemp', '--junitxml', '--timeout', '--override-ini',
          '--config-file', '--ignore-glob'}


def proverit_pytest(args, cwd, own):
    if any(a in ('--noconftest', '-c', '--config-file', '-o', '--override-ini') or a.startswith(('--config-file=', '--override-ini='))
           for a in args):
        otkaz('pytest с обходом конфига тестов (--noconftest / -c / -o).', R_TESTY,
              'запускать pytest с конфигом проекта; менять конфиг — отдельным заданием от владельца.')
    ign, cel, i = set(), [], 0
    while i < len(args):
        a = args[i]
        if a in ('--ignore', '--deselect') and i + 1 < len(args):
            ign.add(norm(args[i + 1].split('::')[0], cwd))
            i += 2
            continue
        if a.startswith(('--ignore=', '--deselect=')):
            ign.add(norm(a.split('=', 1)[1].split('::')[0], cwd))
        elif a in PT_ARG:
            i += 2
            continue
        elif not a.startswith('-'):
            cel.append(norm(a.split('::')[0], cwd))
        i += 1
    cel = cel or [cwd]
    for c in cel:
        for r in {own, MAIN, koren(c) or own}:
            for zt in K['zakrytye_testy']:
                zp = os.path.join(r, zt)
                if pod(zp, c) and zp not in ign and os.path.exists(zp):
                    otkaz('pytest соберёт %s — тест читает позиции 2022+.' % zt,
                          R_DANNYE + '; план 25.10, «своя ошибка дирижёра» (test_core.py)',
                          'указать файлы тестов явно или добавить --ignore=%s.' % zt)


def git_proverka(words, cwd, own):
    i, gcwd = 1, cwd
    while i < len(words) and words[i].startswith('-'):
        if words[i] in ('-C', '-c', '--git-dir', '--work-tree') and i + 1 < len(words):
            if words[i] == '-C':
                gcwd = norm(words[i + 1], gcwd)
            i += 2
        else:
            i += 1
    if i >= len(words):
        return
    sub, args = words[i], words[i + 1:]
    if v_osnovnoy(gcwd, own) and sub not in GIT_RO and not (sub == 'worktree' and args[:1] == ['list']):
        otkaz('git %s в основной копии из рабочей копии агента.' % sub,
              R_KOPIYA + '; поручение 27.10 — checkout проверяющего в основной копии',
              'работать git-ом только в своей копии (cwd = своя рабочая копия).')
    if sub == 'push':
        bad = [a for a in args if a in ('-f', '--force', '--mirror', '--delete', '-d') or a.startswith(('--force', '+'))
               or (not a.startswith('-') and a.startswith(':'))]
        if bad:
            otkaz('git push %s — переписывание или удаление истории.' % ' '.join(bad), R_NEOBR,
                  'обычный git push своей ветки; ошибку исправлять новым коммитом.')
        if own != MAIN and any(a.split(':')[-1] in K['obshchie_vetki'] for a in args if not a.startswith('-')):
            otkaz('push агента прямо в общую ветку.', 'orkestraciya.md абз. 3 (сливает только главная сессия после перепроверки)',
                  'push своей ветки и PR.')
    if sub == 'reset' and '--hard' in args:
        otkaz('git reset --hard — необратимая потеря незакоммиченного.', R_NEOBR,
              'новый коммит с исправлением или git revert; свой файл — git checkout -- <файл>.')
    if sub == 'clean':
        otkaz('git clean удаляет неотслеживаемые файлы безвозвратно.', R_NEOBR, 'удалять свои файлы по одному: rm <файл>.')
    if sub in ('checkout', 'restore', 'rm', 'mv'):
        paths = args[args.index('--') + 1:] if '--' in args else [a for a in args if not a.startswith('-')]
        for a in paths:
            for p in raskryt(a, gcwd):
                if zamorozheno(p):
                    otkaz('git %s перезапишет %s (%s).' % (sub, rel(p), zamorozheno(p)),
                          R_ZAM + '; план 26.10 10:30 — `git checkout origin/... -- scripts/analysis/` в скрипте запуска')


def proverka_bash(cmd, cwd, own):
    segs, hd = razbor(cmd)
    ecwd, eenv = cwd, {}
    for seg in segs:
        env, words, redir = segment(seg)
        if not words:
            eenv.update(env)
            continue
        c0 = os.path.basename(words[0])
        if c0 == 'cd':
            ecwd = norm(words[1], ecwd) if len(words) > 1 else os.path.expanduser('~')
            continue
        if c0 == 'export':
            for w in words[1:]:
                if '=' in w:
                    k, v = w.split('=', 1)
                    eenv[k] = v
            continue
        allenv = dict(eenv, **env)
        redir_p = [norm(r, ecwd) for r in redir if r not in ('/dev/null', '/dev/stderr', '/dev/stdout')]

        # 1. правка файлов оболочкой
        if hd and (redir_p or c0 == 'tee'):
            otkaz('heredoc с записью в файл.', R_PRAVKI + '; план 25.10 — «своя ошибка СD2: правка s9.py через heredoc»',
                  'Write/Edit; heredoc без записи в файл (git commit -F -) разрешён.')
        if c0 in ('sed', 'gsed', 'perl') and any(
                a == '--in-place' or a.startswith('--in-place') or (re.match(r'^-[A-Za-z]*i', a) and not a.startswith('--'))
                for a in words[1:]):
            otkaz('%s -i — правка файла на месте.' % c0, R_PRAVKI, 'Edit.')
        if c0 == 'tee' and [a for a in words[1:] if not a.startswith('-') and a != '/dev/null']:
            otkaz('tee в файл.', R_PRAVKI, 'Write/Edit.')
        if c0 in AVTOR and redir_p:
            otkaz('%s > файл — запись текста оболочкой.' % c0, R_PRAVKI, 'Write/Edit; вывод программ (pytest, скрипты) в файл — можно.')

        # 2. необратимое
        if c0 == 'rm' and any(a == '--recursive' or re.match(r'^-[A-Za-z]*[rR]', a) for a in words[1:]):
            otkaz('рекурсивное удаление (rm -r/-rf) — необратимо; разрешено только как право владельца.',
                  R_NEOBR + '; поручение 27.10 — переполнение диска рабочими копиями (снимать дирижёру git worktree remove)',
                  'свой файл — rm <файл>; папку scratchpad чистит система; рабочую копию — дирижёр через git worktree remove.')
        if c0 == 'find' and ('-delete' in words or any(words[k] in ('-exec', '-execdir') and k + 1 < len(words)
                                                          and os.path.basename(words[k + 1]) == 'rm' for k in range(len(words)))):
            otkaz('find -delete / -exec rm — массовое удаление.', R_NEOBR, 'rm <файл> по одному.')
        if c0 == 'git':
            git_proverka(words, ecwd, own)
        if c0 == 'gh' and words[1:3] == ['repo', 'delete']:
            otkaz('gh repo delete.', R_NEOBR)

        # 3. пути: закрытые, замороженные, основная копия
        pt = pytest_args(words)
        if pt is not None:
            proverit_pytest(pt, ecwd, own)
        cel, args = (py_cel(words) if PYRX.match(c0) or words[0].endswith('.py') else (None, []))
        skript = None
        if cel and cel[0] == 's':
            skript = norm(cel[1], ecwd)
        isk = False  # исключение для скрипта открытия
        if cel and cel[0] in ('s', 'm'):
            astr = ' '.join(args)
            scope_test = bool(re.search(r'--scope(=|\s+)test\b', astr))
            for so in K['skripty_otkrytiya']:
                hit = (skript and rel(skript) == so['put']) or (cel[0] == 'm' and cel[1] == so.get('modul'))
                if not hit:
                    continue
                if not scope_test and any(b in astr for b in so['bezopasno']):
                    break
                sp = skript or os.path.join(koren(ecwd) or own, so['put'])
                why = da_ok(sp)
                if why:
                    otkaz('запуск скрипта открытия %s без безопасного флага (%s): %s.' % (
                        so['put'], ', '.join(so['bezopasno']) or 'нет', why),
                          'pereschet.md п. 9–10 и Ъ 26.10 13:00; канон §4 п. 6 (одно открытие, «да» владельца с датой и SHA)',
                          'владелец кладёт %s: {"v":1,"da":"да","skript":"%s","data":"ДД.ММ.ГГГГ","sha256":"<SHA скрипта>"}.'
                          % (K['da_fail'], so['put']))
                isk = True
                break
            else:
                if scope_test:
                    sp = skript or ''
                    why = da_ok(sp) if sp else 'модуль с --scope test не в списке скриптов открытия'
                    if why:
                        otkaz('--scope test — чтение тестовых лет: %s.' % why, R_DANNYE + '; pereschet.md строки 19–20',
                              'разработка — без --scope test (2018–2021).')
                    isk = True
        if pt is None and cel and cel[0] in ('s', 'm') and own != MAIN and 'FASTBT_MAIN' not in allenv:
            nado = cel[0] == 'm' and cel[1].startswith('fastbt')
            if not nado and skript and skript.endswith('.py'):
                try:
                    with open(skript, encoding='utf-8', errors='ignore') as f:
                        txt = f.read(400000)
                    nado = 'fastbt' in txt or 'FASTBT_MAIN' in txt
                except OSError:
                    nado = False
            if nado and not os.environ.get('FASTBT_MAIN'):
                otkaz('скрипт fastbt из рабочей копии без FASTBT_MAIN — по умолчанию пишет в основную копию.',
                      R_KOPIYA, 'FASTBT_MAIN=<своя копия> python3 ... (данные — симлинками на чтение).')
        fm = allenv.get('FASTBT_MAIN')
        if fm and own != MAIN and '$' not in fm and os.path.realpath(os.path.expanduser(fm)) == MAIN:
            otkaz('FASTBT_MAIN указывает на основную копию.', R_KOPIYA, 'FASTBT_MAIN=<своя копия>.')
        if own != MAIN and v_osnovnoy(ecwd, own) and c0 not in RO and c0 != 'git':
            otkaz('команда «%s» выполняется в основной копии из рабочей копии агента.' % c0, R_KOPIYA,
                  'cd в свою рабочую копию.')

        # кандидаты в пути
        skip, msg_next, pat_done = set(), False, False
        is_grep = c0 in ('grep', 'egrep', 'fgrep', 'rg')
        has_e = any(w in ('-e', '--regexp') or w.startswith('--regexp=') for w in words)
        for k, w in enumerate(words[1:], 1):
            if msg_next:
                skip.add(k)
                msg_next = False
                continue
            if c0 in ('git', 'gh') and (w in MSG_OPTS or w.startswith(('--message=', '--title=', '--body='))):
                msg_next = w in MSG_OPTS
                skip.add(k)
                continue
            if is_grep and w in ('-e', '--regexp'):
                msg_next = True
                continue
            if is_grep and not has_e and not pat_done and not w.startswith('-'):
                pat_done = True
                skip.add(k)
        pisat = []
        if c0 in WRITE:
            nonopt = [w for w in words[1:] if not w.startswith('-')]
            pisat = nonopt[-1:] if c0 in DEST_ONLY else nonopt
        for k, w in enumerate(words):
            if k in skip:
                continue
            for x in kandidaty(w):
                if '/' not in x and not any(c in x for c in '*?[') and not os.path.exists(os.path.join(ecwd, x)):
                    continue
                for p in raskryt(x, ecwd):
                    if not isk and zakryto(p):
                        otkaz('путь %s закрыт (тестовые данные ≥ 2022 / test_locked / замок).' % rel(p), R_DANNYE,
                              'читать только разработку 2018–2021; тест открывает только скрипт открытия после «да».')
        for x in pisat + redir:
            if x in ('/dev/null', '/dev/stderr', '/dev/stdout'):
                continue
            for p in raskryt(x, ecwd):
                z = zamorozheno(p)
                if z:
                    otkaz('запись в %s (%s).' % (rel(p), z), R_ZAM if z != 'конфиг тестов' else R_TESTY)
                if v_osnovnoy(p, own):
                    otkaz('запись в основную копию: %s.' % p, R_KOPIYA, 'писать в свою рабочую копию.')
        # рекурсивный grep с выводом строк над закрытыми папками
        if is_grep:
            rek = c0 == 'rg' or any(re.match(r'^-[A-Za-z]*[rR]', w) or w == '--recursive' for w in words[1:])
            tolko_imena = any(re.match(r'^-[A-Za-z]*[lLc]', w) or w in ('--files-with-matches', '--count', '--files')
                              for w in words[1:])
            filtr = any(w.startswith(('--include', '-g', '--glob', '-t', '--type')) for w in words[1:]) or any(
                w.startswith('--exclude-dir=') and 'data' in w for w in words)
            if rek and not tolko_imena and not filtr:
                nonopt = [w for k, w in enumerate(words[1:], 1) if not w.startswith('-') and k not in skip]
                for d in [norm(x, ecwd) for x in nonopt] or [ecwd]:
                    if os.path.isdir(d) and predok_zakrytogo(d, own):
                        otkaz('рекурсивный поиск со строками по %s захватит data/ и strat_test/ (тест ≥ 2022).' % (rel(d) or '.'),
                              R_DANNYE, 'добавить --include=*.py (или *.md), либо -l (только имена файлов), либо сузить путь.')


# ---------- события ----------

def ctx(inp):
    cwd = os.path.realpath(inp.get('cwd') or os.getcwd())
    own = koren(cwd) or cwd
    return cwd, own


def pre(inp):
    tool, ti = inp.get('tool_name', ''), inp.get('tool_input') or {}
    cwd, own = ctx(inp)
    if tool == 'Bash':
        proverka_bash(ti.get('command', ''), cwd, own)
        return
    p = ti.get('file_path') or ti.get('notebook_path') or ti.get('path')
    p = norm(p, cwd) if p else cwd
    if tool in ('Read', 'Grep', 'Glob') and zakryto(p):
        otkaz('%s закрытого пути %s.' % (tool, rel(p)), R_DANNYE,
              'тест открывает только скрипт открытия после «да» владельца.')
    if tool == 'Grep' and (ti.get('output_mode') == 'content') and os.path.isdir(p) and predok_zakrytogo(p, own):
        g, t = (ti.get('glob') or ''), (ti.get('type') or '')
        kod = re.search(r'\.(py|md|sh|ya?ml|toml|txt|ini|cfg|rs|ipynb)\}?$', g) or t in ('py', 'md', 'sh', 'yaml', 'toml', 'rust')
        if not kod:
            otkaz('Grep со строками по %s без фильтра захватит data/ и strat_test/ (тест ≥ 2022).' % (rel(p) or '.'),
                  R_DANNYE, 'glob "*.py"/"*.md" или type, либо output_mode files_with_matches, либо узкий path.')
    if tool in ('Edit', 'Write', 'MultiEdit', 'NotebookEdit'):
        if zakryto(p):
            otkaz('запись в закрытый путь %s.' % rel(p), R_DANNYE)
        z = zamorozheno(p)
        if z:
            otkaz('правка %s (%s).' % (rel(p), z), R_ZAM if z != 'конфиг тестов' else R_TESTY,
                  'новая версия — новым файлом (main_v2.yaml и т. п.); изменение замка — только владелец.')
        if v_osnovnoy(p, own):
            otkaz('запись в основную копию из рабочей копии агента: %s.' % p, R_KOPIYA,
                  'писать в свою рабочую копию %s.' % own)


def klyuch(inp):
    a = inp.get('agent_id') or 'glavnaya'
    return re.sub(r'[^A-Za-z0-9_.-]', '_', '%s__%s' % (inp.get('session_id', 'bez-sid'), a))


def sost(key, new=None):
    p = os.path.join(AUD, 'sessii', key + '.json')
    if new is None:
        try:
            with open(p, encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError):
            return {'v': 1, 'tronuto': [], 'korni': [], 'stop_blok': 0, 'proveren': ''}
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(new, f, ensure_ascii=False)
    os.replace(tmp, p)


def zhurnal(rec):
    os.makedirs(AUD, exist_ok=True)
    with open(os.path.join(AUD, 'audit.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def testy_modulya(p):
    import glob
    r = koren(p)
    if not r or not p.endswith('.py'):
        return []
    b = os.path.basename(p)
    if b.startswith('test_'):
        return [] if zakrytyy_test(p) else [p]
    if '/tests/' in p or b == 'conftest.py':
        return []
    stem = b[:-3]
    out = set()
    pats = ['tests/test_%s*.py', 'tests/*/test_%s*.py', 'fastbt/tests/test_%s*.py']
    if len(stem) >= 4:  # соглашение проекта: tests/test_s9_<модуль>*.py
        pats += ['tests/test_*_%s*.py', 'fastbt/tests/test_*_%s*.py']
    for pat in pats:
        out.update(glob.glob(os.path.join(r, pat % stem)))
    out.update(glob.glob(os.path.join(os.path.dirname(p), 'tests', 'test_%s*.py' % stem)))
    return sorted(x for x in out if not zakrytyy_test(x))


def pytest_run(files, cwd, timeout):
    import subprocess
    try:
        r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '--no-header', '-p', 'no:cacheprovider', *files],
                           cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           preexec_fn=lambda: os.nice(10))
        return r.returncode in (0, 5), r.returncode, (r.stdout + r.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        return False, -1, 'тесты не уложились в %d с (процесс снят)' % timeout


def post(inp):
    tool, ti = inp.get('tool_name', ''), inp.get('tool_input') or {}
    cwd, own = ctx(inp)
    key = klyuch(inp)
    rec = {'v': 1, 't': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'sid': inp.get('session_id'),
           'agent': inp.get('agent_type') or inp.get('agent_id') or 'glavnaya', 'tool': tool, 'cwd': cwd}
    msgs = []
    if tool == 'Bash':
        rec['cmd'] = (ti.get('command') or '')[:300]
        zhurnal(rec)
    else:
        p = norm(ti.get('file_path') or ti.get('notebook_path') or '', cwd)
        rec['put'], rec['sha256'] = p, sha_file(p)
        zhurnal(rec)
        if p.endswith('.py') and os.path.isfile(p):
            try:
                with open(p, encoding='utf-8') as f:
                    compile(f.read(), p, 'exec')
            except SyntaxError as e:
                msgs.append('py_compile: %s:%s: %s' % (rel(p), e.lineno, e.msg))
            if testy_modulya(p):
                st = sost(key)
                r = koren(p)
                if p not in st['tronuto']:
                    st['tronuto'].append(p)
                if r and r not in st['korni']:
                    st['korni'].append(r)
                sost(key, st)
                if not msgs:
                    fon(key, p)
    bg = os.path.join(AUD, 'fon')
    if os.path.isdir(bg):
        for n in sorted(os.listdir(bg)):
            if n.startswith(key) and n.endswith('.json'):
                q = os.path.join(bg, n)
                try:
                    with open(q, encoding='utf-8') as f:
                        d = json.load(f)
                except (OSError, ValueError):
                    continue
                if not d.get('ok') and not d.get('soobshcheno'):
                    msgs.append('тесты %s упали (код %s):\n%s' % (' '.join(rel(x) for x in d['files']), d['code'], d['tail']))
                    d['soobshcheno'] = True
                    with open(q, 'w', encoding='utf-8') as f:
                        json.dump(d, f, ensure_ascii=False)
    if msgs:
        sys.stderr.write(PREF + 'ПРОВЕРКА ПОСЛЕ ПРАВКИ:\n' + '\n'.join(msgs) +
                         '\nПравило: polnota.md п. 8 (разбор до причины); testy.md п. 1.\n')
        sys.exit(2)


def fon(key, p):
    """Тесты модуля — в фоне (PostToolUse не ждёт); итог — в .claude/audit/fon/<ключ>__<модуль>.json."""
    import subprocess
    d = os.path.join(AUD, 'fon')
    os.makedirs(d, exist_ok=True)
    pid = os.path.join(d, key + '.pid')
    try:
        with open(pid) as f:
            os.kill(int(f.read().strip()), 0)
        return  # прогон этой сессии уже идёт — второй не запускаем (Stop всё равно прогонит)
    except (OSError, ValueError):
        pass
    pr = subprocess.Popen([sys.executable, os.path.abspath(__file__), 'fon', key, p],
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          start_new_session=True, cwd=koren(p) or os.path.dirname(p))
    with open(pid, 'w') as f:
        f.write(str(pr.pid))


def fon_run(key, p):
    files = testy_modulya(p)
    ok, code, tail = pytest_run(files, koren(p) or os.path.dirname(p), K['test_timeout_post'])
    name = re.sub(r'[^A-Za-z0-9_.-]', '_', rel(p))
    with open(os.path.join(AUD, 'fon', '%s__%s.json' % (key, name)), 'w', encoding='utf-8') as f:
        json.dump({'v': 1, 'ok': ok, 'code': code, 'tail': tail, 'files': files, 't': time.time()}, f, ensure_ascii=False)
    try:
        os.remove(os.path.join(AUD, 'fon', key + '.pid'))
    except OSError:
        pass


GATE = re.compile(r'^##\s*(G[\w.-]*)', re.M)


def pending(path):
    try:
        with open(path, encoding='utf-8') as f:
            txt = f.read()
    except OSError:
        return []
    ab = set(re.findall(r'^ABANDON:\s*(G[\w.-]*)', txt, re.M))
    out, cur = [], None
    for ln in txt.splitlines():
        m = GATE.match(ln)
        if m:
            cur = m.group(1)
        elif re.match(r'^EVIDENCE:\s*pending\b', ln.strip(), re.I) and cur and cur not in ab:
            out.append(cur)
    return out


def stop(inp):
    cwd, own = ctx(inp)
    key = klyuch(inp)
    st = sost(key)
    korni = list(dict.fromkeys(st['korni'] + [own]))
    probl = []
    for r in korni:
        g = os.path.join(r, 'GATES.md')
        pe = pending(g)
        if pe:
            probl.append('%s: EVIDENCE: pending у %s — закрыть (CHECK, вывод в EVIDENCE) или ABANDON: <id> <причина>.'
                         % (g, ', '.join(pe)))
    files, shas = [], []
    for p in st['tronuto']:
        for t in testy_modulya(p):
            if t not in files:
                files.append(t)
        shas.append(sha_file(p) or '-')
    stamp = hashlib.sha256('|'.join(st['tronuto'] + shas).encode()).hexdigest()
    if files and st.get('proveren') != stamp:
        by_root = {}
        for t in files:
            by_root.setdefault(koren(t) or own, []).append(t)
        ok_all = True
        for r, fl in by_root.items():
            ok, code, tail = pytest_run(fl, r, K['test_timeout_stop'])
            if not ok:
                ok_all = False
                probl.append('тесты затронутых модулей упали (код %s):\n%s' % (code, tail))
        if ok_all:
            st['proveren'] = stamp
    if probl:
        st['stop_blok'] = st.get('stop_blok', 0) + 1
        sost(key, st)
        zhurnal({'v': 1, 't': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'sid': inp.get('session_id'),
                 'sobytie': 'stop_blok', 'n': st['stop_blok'], 'probl': [x[:200] for x in probl]})
        if st['stop_blok'] > K['stop_povtory']:
            print(json.dumps({'systemMessage': PREF + 'остановка пропущена после %d отказов подряд (orkestraciya.md: '
                              'не более 3 повторов). Отчёт — только «передача» с перечнем незакрытого.' % K['stop_povtory']},
                             ensure_ascii=False))
            return
        sys.stderr.write(PREF + 'ОСТАНОВКА ЗАПРЕЩЕНА (%d из %d):\n- ' % (st['stop_blok'], K['stop_povtory']) +
                         '\n- '.join(probl) + '\nПравило: polnota.md п. 10, 12 (готово при незакрытых гейтах запрещено); '
                         'testy.md п. 1.\n')
        sys.exit(2)
    st['stop_blok'] = 0
    sost(key, st)


def vetka(root):
    try:
        g = os.path.join(root, '.git')
        if os.path.isfile(g):
            with open(g) as f:
                g = f.read().split('gitdir:', 1)[1].strip()
        with open(os.path.join(g, 'HEAD')) as f:
            h = f.read().strip()
        return h.split('refs/heads/', 1)[1] if 'refs/heads/' in h else h[:12]
    except (OSError, IndexError):
        return 'bez-vetki'


def tekst_soobshcheniya(m):
    c = (m.get('message') or {}).get('content')
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return '\n'.join(x.get('text', '') for x in c if isinstance(x, dict) and x.get('type') == 'text')
    return ''


def precompact(inp):
    cwd, own = ctx(inp)
    zadacha, resh, blok = '', [], []
    try:
        with open(inp.get('transcript_path') or '', encoding='utf-8') as f:
            for ln in f:
                try:
                    m = json.loads(ln)
                except ValueError:
                    continue
                t = tekst_soobshcheniya(m)
                if not zadacha and m.get('type') == 'user' and t.strip():
                    zadacha = t.strip()[:1200]
                if m.get('type') == 'assistant':
                    for s in t.splitlines():
                        s = s.strip()
                        if s.startswith(('↺', 'Ъ', '≡')) or '↺ —' in s:
                            resh.append(s[:300])
                        elif s.startswith('❓') or 'БЛОКЕР' in s.upper():
                            blok.append(s[:300])
    except OSError:
        pass
    st = sost(klyuch(inp))
    g = os.path.join(own, 'GATES.md')
    v = vetka(own)
    os.makedirs(HANDOFF, exist_ok=True)
    p = os.path.join(HANDOFF, '%s__%s.md' % (re.sub(r'[^A-Za-z0-9_.-]', '_', v), time.strftime('%Y%m%d-%H%M%S')))
    with open(p, 'w', encoding='utf-8') as f:
        f.write('# Передача (PreCompact, %s)\n\nВетка: %s · копия: %s · сессия: %s\n\n## Задача (первое сообщение)\n%s\n\n'
                '## Решения и отступления\n%s\n\n## Блокеры и вопросы\n%s\n\n## Незакрытые гейты (%s)\n%s\n\n'
                '## Затронутые файлы с тестами\n%s\n' % (
                    time.strftime('%d.%m.%Y %H:%M'), v, own, inp.get('session_id'), zadacha or '—',
                    '\n'.join('- ' + x for x in resh[-15:]) or '—', '\n'.join('- ' + x for x in blok[-10:]) or '—',
                    g, ', '.join(pending(g)) or ('нет' if os.path.exists(g) else 'GATES.md нет'),
                    '\n'.join('- ' + rel(x) for x in st['tronuto']) or '—'))


def start(inp):
    cwd, own = ctx(inp)
    v = vetka(own)
    pre_ = re.sub(r'[^A-Za-z0-9_.-]', '_', v) + '__'
    out = []
    if os.path.isdir(HANDOFF):
        c = sorted(n for n in os.listdir(HANDOFF) if n.startswith(pre_))
        if c:
            with open(os.path.join(HANDOFF, c[-1]), encoding='utf-8') as f:
                out.append('## Последняя передача ветки %s (%s)\n' % (v, c[-1]) + ''.join(f.readlines()[:80]))
    j = os.path.join(own, 'docs', 'kanon_v23', 'zhurnal_resheniy.md')
    if os.path.exists(j):
        with open(j, encoding='utf-8') as f:
            ln = f.readlines()
        out.append('## Журнал решений владельца (последние 25 строк, %s)\n' % rel(j) + ''.join(ln[-25:]))
    import shutil
    import subprocess
    free = shutil.disk_usage(MAIN).free / 2 ** 30
    if free < K.get('disk_min_gb', 5):
        out.append(PREF + 'ДИСК: свободно %.1f ГБ (< %s). Рабочие копии не создавать; дирижёру — git worktree remove '
                   'законченных (план 25.10 ~19:30).' % (free, K.get('disk_min_gb', 5)))
    try:
        ps = subprocess.run(['ps', '-o', 'pid=,etime=,command=', '-U', str(os.getuid())], capture_output=True,
                            text=True, timeout=2).stdout.splitlines()
        dolgie = [s.strip()[:160] for s in ps if 'python' in s and 'sdf_hook' not in s
                  and re.match(r'\s*\d+\s+(\d+-\d\d:\d\d:\d\d|\d\d:\d\d:\d\d|[3-5]\d:\d\d)\s', s)]
        if dolgie:
            out.append(PREF + 'процессы python дольше 30 мин (план 27.10: зависший процесс 54 мин) — проверить и снять по pid:\n'
                       + '\n'.join(dolgie[:10]))
    except (OSError, subprocess.SubprocessError):
        pass
    if out:
        print('\n\n'.join(out))


def main():
    ev = sys.argv[1] if len(sys.argv) > 1 else ''
    if ev == 'fon':
        fon_run(sys.argv[2], sys.argv[3])
        return
    try:
        inp = json.load(sys.stdin)
    except ValueError:
        inp = {}
    {'pre': pre, 'post': post, 'stop': stop, 'precompact': precompact, 'start': start}[ev](inp)


if __name__ == '__main__':
    main()
