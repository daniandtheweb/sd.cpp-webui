import check_requirements as cr


def test_parse(tmp_path):
    f = tmp_path / 'r.txt'
    f.write_text(
        '# comment\n\ngradio==6.31.0\nfoo[extra]>=1.2,<2 ; python_version>"3"\n'
        '-r other.txt\nbar\n')
    assert cr.parse(str(f)) == [
        ('gradio', [('==', '6.31.0')]),
        ('foo', [('>=', '1.2'), ('<', '2')]),
        ('bar', []),
    ]


def test_version_operators():
    assert cr._meets('6.31.0', '==', '6.31.0')
    assert not cr._meets('6.19.0', '==', '6.31.0')
    assert cr._meets('6.31', '==', '6.31.0')
    assert cr._meets('2.0', '>=', '1.9')
    assert not cr._meets('1.10', '<', '1.9')
    assert cr._meets('1.4.2', '~=', '1.4')
    assert not cr._meets('2.0', '~=', '1.4')
    assert cr._meets('6.31.1', '==', '6.31.*')


def test_check_reports_missing_wrong_and_ok(tmp_path):
    f = tmp_path / 'r.txt'
    f.write_text('sdcpp-fake-pkg==1\npip>=0\npip==0.0.1\n')
    ok, msgs = zip(*cr.check(str(f)))
    assert ok == (False, True, False)
    assert 'not installed' in msgs[0]
    assert 'needs ==0.0.1' in msgs[2]


def test_main_exit_codes(tmp_path, capsys):
    good = tmp_path / 'good.txt'
    good.write_text('pip\n')
    bad = tmp_path / 'bad.txt'
    bad.write_text('sdcpp-fake-pkg\n')
    assert cr.main(['x', str(good)]) == 0
    assert cr.main(['x', str(bad)]) == 1
    assert '[!!] sdcpp-fake-pkg' in capsys.readouterr().out
    assert cr.main(['x', str(tmp_path / 'missing.txt')]) == 1
