**********************************
API Reference
**********************************

Documents
=========

Manage document inventories, their versions, and their various stages of processing.

.. autoclass:: ober.documents.DocumentDatabase
	:members:
	:undoc-members:
	:show-inheritance:
	
.. autoclass:: ober.documents.BatchStats
	:members:
	:undoc-members:
	:show-inheritance:
	
Tokens
======

Store token inventories and their vectors. Compute similarities and export token similarity graph.

.. autoclass:: ober.tokens.TokenDatabase
	:members:
	:undoc-members:
	:show-inheritance:
	
Senses
======

Store sense inventories and their vectors.

.. autoclass:: ober.senses.SenseDatabase
	:members:
	:undoc-members:
	:show-inheritance:

Phrases
=======

Combine tokens into common phrases in preparation for Word2Vec training.

.. autoclass:: ober.models.Phrases
	:members:
	:undoc-members:
	:show-inheritance:
	
Word2Vec
========

Word2Vec model for training token vectors in a token inventory.

.. autoclass:: ober.models.Word2Vec
	:members:
	:undoc-members:
	:show-inheritance: