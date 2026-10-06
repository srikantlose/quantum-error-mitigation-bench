from qem.seeds import derive_seed


def test_derive_seed_is_deterministic():
    assert derive_seed("circuit", 4, 2, 0) == derive_seed("circuit", 4, 2, 0)
    assert derive_seed("sim", 2, 4, "low", 3, "base") == derive_seed("sim", 2, 4, "low", 3, "base")


def test_different_parts_give_different_seeds():
    seeds = {
        derive_seed("circuit", 4, 2, 0),
        derive_seed("circuit", 4, 2, 1),
        derive_seed("circuit", 2, 4, 0),
        derive_seed("sim", 4, 2, "low", 0, "base"),
        derive_seed("sim", 4, 2, "low", 0, "fold3"),
        derive_seed("sim", 4, 2, "low", 0, "fold5"),
        derive_seed("sim", 4, 2, "low", 0, "cal"),
        derive_seed("sim", 4, 2, "low", 0, "cal_tensored"),
        derive_seed("qml_split", 0),
        derive_seed("qml_pca", 0),
        derive_seed("qml_init", 4, 2, 0),
        derive_seed("alloc", 4, 2, "low", 0, 1, "uniform", 3),
    }
    assert len(seeds) == 12


def test_seed_range():
    for i in range(2000):
        s = derive_seed("range-check", i)
        assert isinstance(s, int)
        assert 0 <= s < 2**31 - 1
