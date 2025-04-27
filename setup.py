from setuptools import setup

setup(name='heig',
      version='1.2.0',
      description='Highly Efficient Imaging Genetics (HEIG)',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/Zhiwen-Owen-Jiang/heig',
      author='Zhiwen Jiang',
      author_email='zhiwenowenjiang@gmail.com',
      license='GPLv3',
      packages=['heig'],
      scripts=['heig.py'],
      python_requires=">=3.11",
      install_requires=[
          'numpy==1.26.4',
          'pandas==2.2.2',
          'nibabel==5.2.1',
          'scipy==1.11.4',
          'scikit-learn==1.4.2',
          'bitarray==2.9.2',
          'h5py==3.11.0',
          'numexpr==2.10.0',
          'tqdm==4.66.4',
          'pyarrow==17.0.0',
          'filelock==3.15.4',
          'hail==0.2.130',
          'numba==0.61.0'
      ])

