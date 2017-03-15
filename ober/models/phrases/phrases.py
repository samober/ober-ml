import gensim
import os
import json
import codecs
import argparse

from ober.documents import DocumentDatabase
from ober.data import VersionedFile

class Phrases:
    """
    
    Creates a model for finding the most common token pairs and combining them.
    
    For example, the sentence:
    
    ``the``, ``man``, ``went``, ``to``, ``buy``, ``some``, ``ice``, ``cream``, ``.``

    would turn into:
    
    ``the``, ``man``, ``went``, ``to``, ``buy``, ``some``, ``ice_cream``, ``.``
    

    **Usage:**
        
    .. code-block:: python   
   
        from ober.documents import DocumentDatabase
        from ober.models.phrases import Phrases
        
        # load existing model
        phrases = Phrases.load(db_path, version)
        # or create a new one
        phrases = Phrases("data/phrases/bigram", version=1)
        
        # train on sentences
        phrases.train()
        
        # save model
        phrases.save()
        
        # create document databases
        from_document_database = DocumentDatabase.load()
        to_document_database = DocumentDatabase.load(version=from_document_database.version + 1)
        
        # phrase documents
        phrases.phrase_documents(from_document_database, to_document_database)
    
    """
    
    def __init__(self, db_path, version=None):
        self.db_path = db_path
        self.version = version
        
        # create versioned file system
        self.file_base = None
        self._load_file_system(version=self.version)

        self.batches_trained = 0
        self.model = None
        
    def _load_file_system(self, version=None, new_version=False):
        # create versioned file system
        self.file_base = VersionedFile(self.db_path)
        self.version = version
        # if version is None get the latest
        if not self.version:
            self.version = self.file_base.get_latest_version()
        # if there is no current version or new_version, create a new one
        if new_version or self.version == 0:
            self.version = self.file_base.create_latest_version()
            
    def _get_info_file(self):
        return self.file_base.get_file_path(self.version, "info.json")
        
    def _get_model_file(self):
        return self.file_base.get_file_path(self.version, "phrases.model")
        
    def _check_model_file(self):
        return os.path.exists(self._get_model_file())
        
    def enable_fast_model(self):
        """
        Convert model to lightweight model for faster phrasing.
        """
        self.model = gensim.models.phrases.Phraser(self.model)
        
    def train(self, document_database):
        """
        
        Iterates through sentences from the ``DocumentDatabase`` and updates phraser model.
        
        :param document_database: A ``DocumentDatabase`` to use for training.
        :type document_database: ``DocumentDatabase``
        
        """
        # create model
        self.model = gensim.models.Phrases(document_database.get_sentences(), threshold=15.0)
        self.batches_trained = document_database.num_batches()
        
    def phrase_documents(self, from_document_database, to_document_database):
        """
        
        Phrase each sentence of each document in the document inventory and export.
        
        :param from_document_database: The ``DocumentDatabase`` to phrase documents from.
        :type from_document_database: ``DocumentDatabase``
        :param to_document_database: The ``DocumentDatabase`` to save the phrased documents to.
        :type to_document_database: ``DocumentDatabase``
        
        """
        # enable fast mode for faster phrasing
        self.enable_fast_model()
        # phrase
        documents = from_document_database.get_documents()
        to_document_database.add_documents(self._phrase_documents_iter(documents))
        
    def _phrase_documents_iter(self, documents):
        """
        
        Iterates over `documents` and returns phrases documents.
        
        :param documents: The documents to phrase.
        :type documents: list[dict]
        :return: An iterator of JSON documents.
        :rtype: iter[dict]
        
        """
        for document in documents:
            for paragraph in document["paragraphs"]:
                for sentence in paragraph["sentences"]:
                    sentence["tokens"] = self.model[sentence["tokens"]]
            yield document
        
    def save(self, db_path=None, new_version=True):
        """
        
        Saves this ``Phrases`` model to disk.
        
        :param db_path: `(optional)` Path to the phrase inventory.
        :type db_path: str
        :param new_version: `(optional)` If True, increment the current version and export as the latest version.
        :type version: bool
        
        """
        # if there is a new database path, update it and reload the file system
        # if it is just a new version, reload the file system
        if db_path is not None and self.db_path != db_path:
            self.db_path = db_path
            # reload the file system
            self._load_file_system(new_version=new_version)
        elif new_version:
            # reload the file system
            self._load_file_system(version=self.version, new_version=True)
            
        # get the file paths
        info_file = self._get_info_file()
        model_file = self._get_model_file()
        # save info as json
        with codecs.open(info_file, "wb", "utf-8") as outfile:
            outfile.write(json.dumps({ 
                "batches_trained": self.batches_trained }))
        # save model
        self.model.save(model_file)
        
    @staticmethod
    def load(db_path, version=None, fast_model=False):
        """
        
        Loads a ``Phrases`` model from file.
        
        :param db_path: The path to the phrase inventory.
        :type db_path: str
        :param version: The version of the phrase model to load.
        :type version: int
        :param fast_model: If True, enable fast phrasing (will disable training).
        :type fast_model: bool
        :return: ``Phrases`` object for the phrase model.
        :rtype: ``ober.models.phrases.Phrases``
        
        """
        # create the model
        model = Phrases(db_path, version)
        
        # get the path to the info file
        info_file = model._get_info_file()
        # read json
        info_json = json.loads(codecs.open(info_file, "rb", "utf-8").read())
        
        # get phrase model file
        model_file = model._get_model_file()
        # load the model
        model.model = gensim.models.Phrases.load(model_file)
        if fast_model:
            # enable fast model
            model.enable_fast_model()
            
        # copy over json information
        model.batches_trained = info_json["batches_trained"]
        
        return model
        
    @staticmethod
    def get_latest_version(db_path):
        """
        
        Returns the latest version number for the ``Phrases`` model at the database path.
        
        :param db_path: The path to the ``Phrases`` model directory.
        :type db_path: str
        :return: The version id for the latest version.
        :rtype: int 
        
        """
        # create a file system
        file_base = VersionedFile(db_path)
        return file_base.get_latest_version()
            
def main():
    parser = argparse.ArgumentParser(description="Tool for finding common phrases and combining tokens of a document inventory")
    parser.add_argument("--model_path", help="path to the phrases model", default="data/phrases/bigrams")
    parser.add_argument("--model_version", help="version of the phrases model to load (default is new model)", type=int)
    parser.add_argument("--input_documents_version", help="version of documents inventory to train/phrase from", type=int)
    parser.add_argument("--documents_db_path", help="path to the document inventory", default="data/documents")
    parser.add_argument("--no_train", help="do not train the model", action="store_true")
    parser.add_argument("--no_phrase", help="do not phrase any documents", action="store_true")
    parser.add_argument("--output_documents_version", help="version of the documents inventory to phrase to", type=int)
    args = parser.parse_args()
    
    model_path = args.model_path
    model_version = args.model_version
    documents_db_path = args.documents_db_path
    
    # make sure if model_version is None that no_train is False (you must train on a new model)
    if model_version is None and args.no_train:
        print "You must train a new model. Set --no_train to False or specify a --model_version."
        exit()
    
    # create the input document database
    input_document_database = DocumentDatabase.load(documents_db_path, args.input_documents_version)
    
    # create phrases model
    model = None
    if model_version:
        print "Loading model ..."
        model = Phrases.load(model_path, model_version)
    else:
        # create empty model
        model = Phrases(model_path)
    
    if not args.no_train:
        # train
        print "Training ..."
        model.train(input_document_database)
        # save model with a new version
        print "Saving model ..."
        model.save(new_version=True)
        print "Successfully saved version %d at %s ..." % (model.version, model.db_path)
    
    if not args.no_phrase:
        # get the right output version
        output_documents_version = args.output_documents_version
        if not output_documents_version:
            output_documents_version = input_document_database.version + 1
            
        # create the output document database
        output_document_database = DocumentDatabase.load(documents_db_path, output_documents_version)
        
        print "Phrasing ..."
        model.phrase_documents(input_document_database, output_document_database)
    
if __name__ == "__main__":
    main()