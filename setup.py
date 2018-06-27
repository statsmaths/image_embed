from setuptools import setup
from setuptools import find_packages

setup(name='Image Embedding',
      version='0.1.0',
      description='Simple API to map image files into embedding spaces',
      author='Taylor Anold',
      author_email='taylor.arnold@acm.org',
      url='https://github.com/statsmaths/image_embed',
      license='GPL-2',
      install_requires=['numpy>=1.14.0',
                        'keras>=2.1.4',
                        'scipy>=1.0.0',
                        'h5py>=2.7.1',
                        'umap-learn'],
      extras_require={
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov'],
      },
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v2 or later'
          '(GPLv2+)',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Image Recognition',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
