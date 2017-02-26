import gensim
import os
import json
import codecs
import argparse

from ober.documents import DocumentDatabase

class Phrases:
    def __init__(self, db_path, version=None, documents_version=None, document_set=None):
        self.db_path = db_path
        self.version = version
        
        self.documents_version = documents_version
        self.document_set = document_set
        self.has_trained = False
        
        self.batches_trained = 0
        
        self.model = None
    
    def init_model(self):
        self.model = gensim.models.Phrases(threshold=15.0)
        
    def enable_fast_model(self):
        self.model = gensim.models.phrases.Phraser(self.model)
        
    def train(self):
        document_database = DocumentDatabase.load(version=self.documents_version, document_set=self.document_set)
        # create model
        self.model = gensim.models.Phrases(document_database.get_sentences(), threshold=15.0)
        self.batches_trained = document_database.num_batches()
        
    def phrase_documents(self, documents_version, documents_set):
        self.enable_fast_model()
        
        document_database = DocumentDatabase.load(version=self.documents_version, document_set=self.document_set)
        new_document_database = DocumentDatabase(version=documents_version, document_set=documents_set)
        # phrase
        documents = document_database.get_documents()
        new_document_database.add_documents(self.phrase_documents_iter(documents))
        
    def phrase_documents_iter(self, documents):
        for document in documents:
            for paragraph in document["paragraphs"]:
                for sentence in paragraph["sentences"]:
                    sentence["tokens"] = self.model[sentence["tokens"]]
            yield document
        
    def save(self, db_path=None, version=None):
        if not db_path:
            db_path = self.db_path
        if not version:
            version = self.version
        Phrases.ensure_phrases_dir(db_path)
        info_file = Phrases.get_info_file(db_path, version)
        model_file = Phrases.get_model_file(db_path, version)
        # save info as json
        with codecs.open(info_file, "wb", "utf-8") as outfile:
            outfile.write(json.dumps({ 
                "documents_version": self.documents_version, 
                "document_set": self.document_set, 
                "has_trained": self.has_trained, 
                "batches_trained": self.batches_trained }))
        # save model
        self.model.save(model_file)
        
    @staticmethod
    def load(db_path, version, fast_model=False):
        # load info file
        info_file = Phrases.get_info_file(db_path, version)
        # read json
        info_json = json.loads(codecs.open(info_file, "rb", "utf-8").read())
        # get phrase model file
        model_file = Phrases.get_model_file(db_path, version)
        model = Phrases(db_path=db_path, version=version, documents_version=info_json["documents_version"], document_set=info_json["document_set"])
        if fast_model:
            model.model = gensim.models.phrases.Phraser.load(model_file)
        else:
            model.model = gensim.models.Phrases.load(model_file)
        model.has_trained = info_json["has_trained"]
        model.batches_trained = info_json["batches_trained"]
        return model
        
    @staticmethod
    def get_latest_version(db_path):
        max_version = 0
        for f in os.listdir(db_path):
            if f.endswith(".phrases"):
                version = 0
                try:
                    version = int(f.split(".")[0])
                except:
                    version = 0
                if version > max_version:
                    max_version = version
        return max_version
        
    @staticmethod
    def get_model_file(db_path, version):
        return os.path.join(db_path, "%05d.phrases" % version)
        
    @staticmethod
    def get_info_file(db_path, version):
        return os.path.join(db_path, "%05d.info.json" % version)
        
    @staticmethod
    def ensure_phrases_dir(db_path):
        if not os.path.exists(db_path):
            os.makedirs(db_path)
            
def main():
    parser = argparse.ArgumentParser(description="Tool for finding common phrases and combining tokens of a document inventory")
    parser.add_argument("--model_path", help="path to the phrases model", default="data/phrases/bigrams")
    parser.add_argument("--model_version", help="version of the phrases model to load (default is new model)", type=int)
    parser.add_argument("--documents_version", help="version of documents inventory to train/phrase from", type=int)
    parser.add_argument("--document_set", help="document set to train/phrase from", default="raw")
    parser.add_argument("--documents_db_path", help="path to the document inventory", default="data/documents")
    parser.add_argument("--no_train", help="do not train the model", action="store_true")
    parser.add_argument("--no_phrase", help="do not phrase any documents", action="store_true")
    parser.add_argument("--output_documents_version", help="version of the documents inventory to phrase to", type=int)
    parser.add_argument("--output_document_set", help="document set to phrase into", default="bigrams")
    args = parser.parse_args()
    
    model_path = args.model_path
    model_version = args.model_version
    documents_db_path = args.documents_db_path
    documents_version = args.documents_version
    document_set = args.document_set
    if not documents_version:
        documents_version = DocumentDatabase.get_latest_version(documents_db_path, document_set)
    
    # create phrases model
    model = None
    if model_version:
        print "Loading model ..."
        model = Phrases.load(model_path, model_version)
    else:
        model_version = Phrases.get_latest_version(model_path) + 1
        model = Phrases(model_path, version=model_version, documents_version=documents_version, document_set=document_set)
    
    if not args.no_train:
        # train
        print "Training ..."
        model.train()
        # save model
        print "Saving model ..."
        model.save()
    
    if not args.no_phrase:
        output_documents_version = args.output_documents_version
        output_document_set = args.output_document_set
        if not output_documents_version:
            output_documents_version = DocumentDatabase.get_latest_version(documents_db_path, output_document_set) + 1
        
        print "Phrasing ..."
        model.phrase_documents(output_documents_version, output_document_set)
    
if __name__ == "__main__":
    main()