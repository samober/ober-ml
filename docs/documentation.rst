**********************************
Documentation
**********************************

Overview
=========

#TODO

Phrasing with ``ober.models.Phrases``
=============================================================

Bigrams:

.. code-block:: python

	python -m ober.models.phrases.phrases --model_path data/phrases/bigrams --document_set raw --output_document_set bigrams
	
Trigrams:

.. code-block:: python

	python -m ober.models.phrases.phrases --model_path data/phrases/trigrams --document_set bigrams --output_document_set trigrams

Creating/Updating Token Inventories
========================================

.. code-block:: python

	python -m ober.models.word2vec.update_tokens

Exporting Token Similarity Graphs
=====================================

.. code-block:: python

	python -m ober.models.sense2vec.export_graph

Sense Induction with Graph Clustering
=======================================

.. code-block:: python

	python -m ober.models.sense2vec.chinese_whispers

Creating Sense Inventories
==============================

.. code-block:: python

	python -m ober.models.sense2vec.pool_vectors

Word Sense Disambiguation
==============================

#TODO

Working with Skip Thought Vectors
==================================

#TODO