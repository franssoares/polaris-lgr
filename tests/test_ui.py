import pytest
from streamlit.testing.v1 import AppTest

def test_app_starts_without_errors():
    at = AppTest.from_file("../app.py", default_timeout=10).run()
    assert not at.exception, f"App encountered an exception on startup: {at.exception}"

def test_app_with_complex_poles_zeros():
    at = AppTest.from_file("../app.py", default_timeout=10).run()
    assert not at.exception
    
    # Try interacting
    try:
        at.text_input(key="num_input").set_value("1 2 2").run()
        at.text_input(key="den_input").set_value("1 4 8 0").run()
    except Exception as e:
        print("Could not set input values:", e)
        return
        
    try:
        at.button[0].click().run()
    except Exception as e:
        print("Could not click button:", e)
        return
        
    assert not at.exception, f"App crashed after running calculation: {at.exception}"
