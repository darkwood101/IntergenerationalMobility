# IntergenerationalMobility
Optimal Opportunity Allocation and Affirmative Action to Maximize Intergenerational Mobility

## Testing

**TLDR**: `./run-tests.sh` to ensure all tests are passing before you push

To run all unit tests once, you can execute
```
python -m unittest
```
from the main directory. However, our affirmative action model has a lot of
inherent randomness. Thus, it could happen that the tests pass a single run,
even though the code is not correct. If we increase the number of runs,
it gets increasingly more unlikely that incorrect code would pass all the runs.
Therefore, we have an automated testing shell script, `run-test.sh`. Simply run
```
./run-tests.sh
```
from the main directory, and the script will execute all tests 100 times,
reporting if any of them fails.

To add more tests, create a file in the `tests/` directory. Within that file,
you should `import unittest` and declare a class that inherits from
`unittest.TestCase`. Then, each method that you add to this class will represent
a separate test. Here is a sample `test_foo.py`:
```python
class test_foo(unittest.TestCase):
    def test_foo_baz(self):
        f = foo(41)
        self.assertEqual(f.baz(), 42)
        
        f = foo(226)
        self.assertEqual(f.baz(), 227)


    def test_foo_bar(self):
        ...
```
You can also look at existing tests in the `tests/` directory.

Do not modify `tests/test.py`.

Please make sure that all tests are passing before you push.

