def test_julia_set_generation():
    from src.advanced_harmony import JuliaSet
    julia = JuliaSet()
    fractal = julia.generate(100, 100)
    assert fractal.shape == (100, 100)
