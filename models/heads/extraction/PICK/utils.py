from pathlib import Path
from collections import Counter
from torchtext.vocab import Vocab   # Compatibility: https://pypi.org/project/torchtext/


Entities_list = ['SELLER', 'ADDRESS', 'TIMESTAMP', 'TOTAL_COST',]

# Entities_list = ['VAT_amount', 'VAT_amount_val', 'VAT_rate', 'VAT_rate_val', 'account_no', 'address', 'address_val',
#                  'amount_in_words', 'amount_in_words_val', 'bank', 'buyer', 'company_name', 'company_name_val', 'date',
#                  'exchange_rate', 'exchange_rate_val', 'form', 'form_val', 'grand_total', 'grand_total_val', 'no',
#                  'no_val', 'seller', 'serial', 'serial_val', 'tax_code', 'tax_code_val', 'total', 'total_val', 'website']

# Entities_list = ["ticket_num", "starting_station",
#                   "train_num", "destination_station",
#                   "date", "ticket_rates", "seat_category", "name",]


class ClassVocab(Vocab):

    def __init__(self, classes, specials=['<pad>', '<unk>'], **kwargs):
        """
        Convert key to index(stoi), and get key string by index(itos)
        
        Parameters
        ----------
        classes: list or str, key string or entity list
        specials: list, special tokens except <unk> (default: {['<pad>', '<unk>']})
        kwargs:
        """
        cls_list = None
        if isinstance(classes, str):
            cls_list = list(classes)
        if isinstance(classes, Path):
            p = Path(classes)
            if not p.exists():
                raise RuntimeError('Key file is not found')
            with p.open(encoding='utf8') as f:
                classes = f.read()
                classes = classes.strip()
                cls_list = list(classes)
        elif isinstance(classes, list):
            cls_list = classes
        counter = Counter(cls_list)
        self.special_count = len(specials)
        super().__init__(counter, specials=specials, **kwargs)


def entities2iob_labels(entities: list):
    """ Get all Beginning-Inside-Outside string label by entities """
    tags = []
    for e in entities:
        tags.append('B-{}'.format(e))
        tags.append('I-{}'.format(e))
    tags.append('O')
    return tags


keys_vocab_cls         = ClassVocab(Path(__file__).parent / 'keys_vn.txt', specials_first=False)
entities_vocab_cls     = ClassVocab(                      Entities_list  , specials_first=False)
entities_iob_vocab_cls = ClassVocab(  entities2iob_labels(Entities_list) , specials_first=False)


