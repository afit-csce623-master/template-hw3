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
  with testbook('hw2.ipynb', execute=True, timeout=120) as tb:
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
    
    hashes = ['cd9b9ed6e69ab865f6702294a0154fc8', 'd88cae0807b4e82ebcef60de2487fb1f', 'f465bf3ebafdec97662e0ea770f66252']
    
    count = 3
    for i in range(1, count+1):
        val = str(verify_exists(tb, f'df{i}', 1)).encode('utf-8')
        
        assert hashlib.md5(val).hexdigest() == hashes[i-1], \
            f'STEP 1: df{i} does not appear to be loaded correctly. Verify data source, columns, and headers'
   
    
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
        
    split = verify_exists(tb, 'split', 5)
    
    tb.inject(f"""
        {V_HASH}df = pd.read_csv("https://raw.githubusercontent.com/afit-csce623-master/datasets/main/hw2_dataset1.csv", header=0, names=["X1", "X2", "Class"], index_col=0)
        
        {V_HASH}size_array = [50, 100, 200, 300, 400, 500]
        
        for idx, {V_HASH}size in enumerate({V_HASH}size_array): 
            
            {V_HASH}X_train, {V_HASH}X_test, {V_HASH}y_train, {V_HASH}y_test = split({V_HASH}df, {V_HASH}size)
            
            {V_HASH}train_size = np.size({V_HASH}X_train, 0)   
            {V_HASH}test_size = np.size({V_HASH}X_test, 0)   
            {V_HASH}class_0_train_size = int(({V_HASH}y_train == 0).sum()) 
            {V_HASH}class_0_test_size = int(({V_HASH}y_test == 0).sum())
            {V_HASH}class_1_train_size = int(({V_HASH}y_train == 1).sum())
            {V_HASH}class_1_test_size = int(({V_HASH}y_test == 1).sum())  
            
            if {V_HASH}train_size != {V_HASH}size:
                break                   
            if {V_HASH}test_size != (600 - {V_HASH}size):
                break
            if {V_HASH}class_0_train_size != {V_HASH}size / 2:
                break
            if {V_HASH}class_0_test_size != (600 - {V_HASH}size) / 2:
                break
            if {V_HASH}class_1_train_size != {V_HASH}size / 2:
                break                
            if {V_HASH}class_1_test_size != (600 - {V_HASH}size) / 2:
                break
    """)
 
    exec(f'global size; size = tb.ref("{V_HASH}size")')
    exec(f'global train_size; train_size = tb.ref("{V_HASH}train_size")')
    exec(f'global test_size; test_size = tb.ref("{V_HASH}test_size")')
    exec(f'global class_0_train_size; class_0_train_size = tb.ref("{V_HASH}class_0_train_size")')
    exec(f'global class_0_test_size; class_0_test_size = tb.ref("{V_HASH}class_0_test_size")')
    exec(f'global class_1_train_size; class_1_train_size = tb.ref("{V_HASH}class_1_train_size")')
    exec(f'global class_1_test_size; class_1_test_size = tb.ref("{V_HASH}class_1_test_size")')

    assert train_size == size, f'STEP 5: Check split() function. When called with split(df, {size}), the function returns a training set with size {train_size}, but it should be {size}.'
    assert test_size == 600 - size, f'STEP 5: Check split() function. When called with split(df, {size}), the function returns a test set with size {test_size}, but it should be {600 - size}.'
    assert class_0_train_size == size / 2, f'STEP 5: Check split() function. It should return a balanced distribution of classes between the train and test set. When called with split(df, {size}), the function returns a train set with {class_0_train_size} observations in Class 0, but there should be {size / 2} observations. Look closely at the train_test_split function parameters.'
    assert class_0_test_size == (600 - size) / 2, f'STEP 5: Check split() function. It should return a balanced distribution of classes between the train and test set. When called with split(df, {size}), the function returns a test set with {class_0_test_size} observations in Class 0, but there should be {(600 - size) / 2} observations. Look closely at the train_test_split function parameters.'    
    assert class_1_train_size == size / 2, f'STEP 5: Check split() function. It should return a balanced distribution of classes between the train and test set. When called with split(df, {size}), the function returns a train set with {class_1_train_size} observations in Class 1, but there should be {size / 2} observations. Look closely at the train_test_split function parameters.'
    assert class_1_test_size == (600 - size) / 2, f'STEP 5: Check split() function. It should return a balanced distribution of classes between the train and test set. When called with split(df, {size}), the function returns a test set with {class_1_test_size} observations in Class 1, but there should be {(600 - size) / 2} observations. Look closely at the train_test_split function parameters.'

    exec(f'global X_train; X_train = tb.ref("{V_HASH}X_train")')
    exec(f'global X_test; X_test = tb.ref("{V_HASH}X_test")')
    exec(f'global y_train; y_train = tb.ref("{V_HASH}y_train")')
    exec(f'global y_test; y_test = tb.ref("{V_HASH}y_test")')
    
    assert hashlib.md5(str(X_train).encode('utf-8')).hexdigest() == '6a1da022873521ca5d0ebbac75270e5a', f'STEP 5: The X_train set is incorrect. Be sure that you are using a default value of 42 for the random state, and that you are applying it to the train_test_split function.'
    assert hashlib.md5(str(X_test).encode('utf-8')).hexdigest() == '7b66c512b3cba7699f6335c482af0b99', f'STEP 5: The X_test set is incorrect. Be sure that you are using a default value of 42 for the random state, and that you are applying it to the train_test_split function.'
    assert hashlib.md5(str(y_train).encode('utf-8')).hexdigest() == '31dbc506ba8e94324c3d57fb03f0bf00', f'STEP 5: The y_train set is incorrect. Be sure that you are using a default value of 42 for the random state, and that you are applying it to the train_test_split function.'
    assert hashlib.md5(str(y_test).encode('utf-8')).hexdigest() == '4ba37d77563bf2292800119d498c7cda', f'STEP 5: The X_test set is incorrect. Be sure that you are using a default value of 42 for the random state, and that you are applying it to the train_test_split function.'
    
    
def test_step_6(tb):
    try:
        complete = None
        complete = tb.ref('STEP_6_COMPLETE')
    except:
        # STEP_6_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 6: not complete.' 
    
    X_train_1 = verify_exists(tb, 'X_train_1', 6)
    X_test_1 = verify_exists(tb, 'X_test_1', 6)
    y_train_1 = verify_exists(tb, 'X_train_1', 6)
    y_test_1 = verify_exists(tb, 'X_train_1', 6)
    X_train_2 = verify_exists(tb, 'X_train_2', 6)
    X_test_2 = verify_exists(tb, 'X_test_2', 6)
    y_train_2 = verify_exists(tb, 'X_train_2', 6)
    y_test_2 = verify_exists(tb, 'X_train_2', 6)
    X_train_3 = verify_exists(tb, 'X_train_3', 6)
    X_test_3 = verify_exists(tb, 'X_test_3', 6)
    y_train_3 = verify_exists(tb, 'X_train_3', 6)
    y_test_3 = verify_exists(tb, 'X_train_3', 6)
    
    for idx in range(1, 4):
        verify_array_size(tb, f'X_train_{idx}', 200, 6)
        verify_array_size(tb, f'X_test_{idx}', 200, 6)
        verify_array_size(tb, f'y_train_{idx}', 200, 6)
        verify_array_size(tb, f'y_test_{idx}', 200, 6)
        
    hashes = ['1aa9b27990e4d100fba7e31d2afc509f', '8210865bdabd83d7ef494e34b1ce312e', 'f9e26688f350cad3416c31f98e8c284e',
              '4023aa54dc30103ab3fcd9a5c5f287be', 'd06d91df4432b6fd8c3aa26d1b52234f', 'b9f7799b20ea203561339fe9843a6d99',
              '3e931aa9c1dbbc72b4e80252929596c3', '42d47fae9df940fdf8181c7ba49adcac', '78e1de526101dd1e7bf43c407bcdf4f4',
              'cd3848ab651df7a8afe4cafdaa41b800', '1ca8b56672883d36bf34cf97a2419d3c', '3cf2cf25654a64d876a6b537661cfaba']
    
    message = ''
    for idx in range(1, 4):
        verify_hash(tb, f'X_train_{idx}', hashes[(idx-1)*4 + 0], 6, message)
        verify_hash(tb, f'X_test_{idx}', hashes[(idx-1)*4 + 1], 6, message)
        verify_hash(tb, f'y_train_{idx}', hashes[(idx-1)*4 + 2], 6, message)
        verify_hash(tb, f'y_test_{idx}', hashes[(idx-1)*4 + 3], 6, message)
            
    
def test_step_7(tb):
    try:
        complete = None
        complete = tb.ref('STEP_7_COMPLETE')
    except:
        # STEP_7_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 7: not complete.'    

    verify_exists(tb, 'train_classifiers', 7)

    tb.inject(f"""
        np.random.seed(42)
        X = np.random.random((100,2))
        y = np.around(np.random.random(100))
        
        {V_HASH}models1 = train_classifiers(X, y)       
    """)
    
    verify_hash(tb, f'{V_HASH}models1', 'c739dca3a89a2af44ced5b468737afb7', 7, 'train_classifiers is not returning the correct value. Verify that your function is creating three entries in a Python dictionary. The unit test assumes that the order of the entries is {"log", "lda", "qda"}')
    
    
def test_step_8(tb):
    try:
        complete = None
        complete = tb.ref('STEP_8_COMPLETE')
    except:
        # STEP_8_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 8: not complete.' 
    
    verify_exists(tb, 'models1', 8)
    verify_exists(tb, 'models2', 8)
    verify_exists(tb, 'models3', 8)
    
    print_hash(tb, 'models1')
    print_hash(tb, 'models2')
    print_hash(tb, 'models3')
    
    
def test_step_9(tb):
    try:
        complete = None
        complete = tb.ref('STEP_9_COMPLETE')
    except:
        # STEP_9_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 9: not complete.' 
    
    verify_exists(tb, 'predict_probabilities', 9)
    
    tb.inject(f"""
        np.random.seed(42)
        X = np.random.random((100,2))
        y = np.around(np.random.random(100))
        
        {V_HASH}models = train_classifiers(X, y) 
        {V_HASH}predicts_log = predict_probabilities({V_HASH}models, X)['log']
    """)
    
    dim_count = get_array_dim_count(tb, f'{V_HASH}predicts_log')

    assert dim_count == 1, f'STEP 9: predict_probabilities returns an array of {dim_count} columns. It should be a 1d array. Be sure to slice the result of predict_proba to return only the probability an observation is in Class 1.'
    
    verify_hash(tb, f'{V_HASH}predicts', 'ab4cb0c6dc08f92b1140913a4dc1abdd', 9, "The function predict_probabilities does not return the correct value. Verify that you're using predict_proba.")
    
    
def test_step_10(tb):
    try:
        complete = None
        complete = tb.ref('STEP_10_COMPLETE')
    except:
        # STEP_10_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 10: not complete.'      
    
    verify_exists(tb, 'predicts1', 10)
    verify_exists(tb, 'predicts2', 10)    
    verify_exists(tb, 'predicts3', 10)
    
    verify_hash(tb, 'predicts1', '57c9ea12ba7e4c40ef5cf3273222e0f0', 10)
    verify_hash(tb, 'predicts2', '3ea354fc658bf93c9330a048ae8dc91d', 10)
    verify_hash(tb, 'predicts3', 'a3fb3ca4b823f0bb8033e62f45cf9a1c', 10)   
    

def test_step_11(tb):
    try:
        complete = None
        complete = tb.ref('STEP_11_COMPLETE')
    except:
        # STEP_11_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        assert complete, 'STEP 11: not complete.'

    tb.inject(f"""
        {V_HASH}STEP_11_NameError = ''

        np.random.seed(623)
        {V_HASH}columns=['Threshold', 'True Positive', 'False Positive', 'True Negative', 'False Negative', 'Recall', 'Precision', 'False Positive Rate', 'Accuracy', 'F-measure']
        {V_HASH}y_probs = np.random.random(10)        
        {V_HASH}truths = np.around(np.random.random(10))
        {V_HASH}thresholds = [0.25, 0.5, 0.75]
        {V_HASH}threshold_metrics_results = threshold_metrics({V_HASH}truths, {V_HASH}y_probs, {V_HASH}thresholds)
        for index, row in {V_HASH}threshold_metrics_results.iterrows():
            for column in {V_HASH}columns:
                column_str = column.replace(' ','_').replace('-','_').lower()
                try:
                    if {V_HASH}STEP_11_NameError == '':
                        exec('{V_HASH}threshold_metrics_results_' + str(index) + '_' + column_str + ' = ' + str(row[column]))
                except NameError:
                    {V_HASH}STEP_11_NameError = column
    """)

    
    compare = [[0.25, 6.0, 1.0, 0.0, 3.0, 0.6666666666666666, 0.8571428571428571, 1.0, 0.6, 0.75], 
               [0.5, 6.0, 1.0, 0.0, 3.0, 0.6666666666666666, 0.8571428571428571, 1.0, 0.6, 0.75],
               [0.75, 4.0, 1.0, 0.0, 5.0, 0.4444444444444444, 0.8, 1.0, 0.4, 0.5714285714285714]]  
    
    for idx1, threshold in enumerate([0.25, 0.5, 0.75]):
        for idx2, column in enumerate(['Threshold', 'True Positive', 'False Positive', 'True Negative', 'False Negative', 'Recall', 'Precision', 'False Positive Rate', 'Accuracy', 'F-measure']):
            column_str = column.replace(' ','_').replace('-','_').lower()
            
            if DEBUG:
                # sample code to display values
                print(f'{idx1} {threshold} {column_str}: ', 
                      f'{V_HASH}threshold_metrics_results_{idx1}_{column_str}', 
                      tb.ref(f'{V_HASH}threshold_metrics_results_{idx1}_{column_str}'))
            
                # code to display exec call            
                print(f'global val; val = float(tb.ref("{V_HASH}threshold_metrics_results_{idx1}_{column_str}"))')

                
            try:
                exec(f'global val; val = float(tb.ref("{V_HASH}threshold_metrics_results_{idx1}_{column_str}"))')
            except Exception as e:
                assert False, f"STEP 11: Couldn't execute test. Perhaps the {column} calculation has not been implemented or the column is named incorrectly?"
            
            print(f'STEP 11: {column} calculation is not correct')
            assert np.isclose(val,compare[idx1][idx2]), f'STEP 11: {column} calculation is not correct'
        

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
    
    tb.inject(f"""
        np.random.seed(42)
        {V_HASH}X = np.random.random((100,2))
        {V_HASH}y = np.around(np.random.random(100))
        
        {V_HASH}models = train_classifiers({V_HASH}X, {V_HASH}y) 
        {V_HASH}predicts = predict_probabilities({V_HASH}models, {V_HASH}X)
        {V_HASH}thresholds_argmax = thresholds_argmax({V_HASH}y, {V_HASH}predicts)
    """)
    
    print_hash(tb, f'{V_HASH}thresholds_argmax')

    
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

    
def test_step_20(tb):
    try:
        complete = None
        complete = tb.ref('STEP_20_COMPLETE')
    except:
        # STEP_20_COMPLETE constant has been removed, set to true
        complete = True
    finally:
        pass # STEP 20 is optional
