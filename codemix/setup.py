from setuptools import setup

setup(
    name='codemix',
    version='1.0',
    author='Prashant Kodali',
    author_email='prashant.kodali@research.iiit.ac.in',
    description='A toolkit for code-mixed language processing',
    packages=['codemix'],
    install_requires=[
        'statistics',
        'streamlit',
        'htbuilder',
        'IPython',
<<<<<<< HEAD
        'st-annotated-text'
=======
        'st-annotated-text',
        'pandas',
        'stanza',
        'indic-nlp-library',
        'torch',
        'transformers'
>>>>>>> d3f9b6f1c18af985471d67d8ed2a2109dce2a082
    ],
)
