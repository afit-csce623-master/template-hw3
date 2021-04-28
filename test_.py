import pytest
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model
from testbook import testbook
    
V_HASH = 'lkfidhzwpt_'
DEBUG = True

@pytest.fixture(scope='module')
def tb():
  with testbook('hw3.ipynb', execute=True, timeout=120) as tb:
    yield tb

    
def verify_exists(tb, name, num_test_case):
    try:
        val = tb.ref(name)
    except Exception as e:
        if DEBUG:
            print(type(e))
            print(e.args)
            print(e)
        assert False, f'STEP {num_test_case}: {name} does not exist. Ensure that {name} is defined.'
        
    return val


def get_array_size(tb, name):
    tb.inject(f"""
        {V_HASH}val_size = np.size({name},0)
    """)
    
    exec(f'global array_size; array_size = tb.ref("{V_HASH}val_size")')
    return array_size


def get_array_dim_count(tb, name):
    tb.inject(f"""
        {V_HASH}val_size = {name}.ndim
    """)
    exec(f'global array_size; array_size = tb.ref("{V_HASH}val_size")')
    return array_size
    

def verify_array_size(tb, name, size, num_test_case):
    array_size = get_array_size(tb, name)
    assert array_size == size, f'STEP {num_test_case}: {name} is the wrong size. {name} should be {size} observations. Instead, it is {array_size} observations.'
    return array_size


def print_hash(tb, name):
    print(hashlib.md5(str(name).encode('utf-8')).hexdigest())
    
    
def verify_hash(tb, name, hash_string, num_test_case, message = None):
    if message:
        assert hashlib.md5(str(name).encode('utf-8')).hexdigest() == hash_string, f'STEP {num_test_case}: {message}.'
    else:
        assert hashlib.md5(str(name).encode('utf-8')).hexdigest() == hash_string, f'STEP {num_test_case}: {name} has the wrong value.'
    
def test_step_1(tb):
    try:
        complete = None
        complete = tb.ref('STEP_1_COMPLETE')
    except:
        # STEP_1_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 1: not complete.'
   
    
def test_step_2(tb):
    try:
        complete = None
        complete = tb.ref('STEP_2_COMPLETE')
    except:
        # STEP_2_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 2: not complete.'
    
    
def test_step_3(tb):
    try:
        complete = None
        complete = tb.ref('STEP_3_COMPLETE')
    except:
        # STEP_3_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 3: not complete.'

    
def test_step_4(tb):
    try:
        complete = None
        complete = tb.ref('STEP_4_COMPLETE')
    except:
        # STEP_4_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 4: not complete.'

    
def test_step_5(tb):
    try:
        complete = None
        complete = tb.ref('STEP_5_COMPLETE')
    except:
        # STEP_5_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 5: not complete.'  
  

def test_step_6(tb):
    try:
        complete = None
        complete = tb.ref('STEP_6_COMPLETE')
    except:
        # STEP_6_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 6: not complete.' 
            
    
def test_step_7(tb):
    try:
        complete = None
        complete = tb.ref('STEP_7_COMPLETE')
    except:
        # STEP_7_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 7: not complete.'    

    
def test_step_8(tb):
    try:
        complete = None
        complete = tb.ref('STEP_8_COMPLETE')
    except:
        # STEP_8_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 8: not complete.' 

    
def test_step_9(tb):
    try:
        complete = None
        complete = tb.ref('STEP_9_COMPLETE')
    except:
        # STEP_9_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 9: not complete.'     

    
def test_step_10(tb):
    try:
        complete = None
        complete = tb.ref('STEP_10_COMPLETE')
    except:
        # STEP_10_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 10: not complete.'      
    

def test_step_11(tb):
    try:
        complete = None
        complete = tb.ref('STEP_11_COMPLETE')
    except:
        # STEP_11_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 11: not complete.'


def test_step_12(tb):
    try:
        complete = None
        complete = tb.ref('STEP_12_COMPLETE')
    except:
        # STEP_12_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 12: not complete.'
        
    verify_exists(tb, 'thresholds_argmax', 12)
    
    
def test_step_13(tb):
    try:
        complete = None
        complete = tb.ref('STEP_13_COMPLETE')
    except:
        # STEP_13_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 13: not complete.'


def test_step_14(tb):
    try:
        complete = None
        complete = tb.ref('STEP_14_COMPLETE')
    except:
        # STEP_14_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 14: not complete.'


def test_step_15(tb):
    try:
        complete = None
        complete = tb.ref('STEP_15_COMPLETE')
    except:
        # STEP_16_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 15: not complete.'


def test_step_16(tb):
    try:
        complete = None
        complete = tb.ref('STEP_16_COMPLETE')
    except:
        # STEP_16_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 16: not complete.'
    

def test_step_17(tb):
    try:
        complete = None
        complete = tb.ref('STEP_17_COMPLETE')
    except:
        # STEP_17_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 17: not complete.'
    
    
def test_step_18(tb):
    try:
        complete = None
        complete = tb.ref('STEP_18_COMPLETE')
    except:
        # STEP_18_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 18: not complete.' 
    
    
def test_step_19(tb):
    try:
        complete = None
        complete = tb.ref('STEP_19_COMPLETE')
    except:
        # STEP_19_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        pass # STEP 19 is optional
