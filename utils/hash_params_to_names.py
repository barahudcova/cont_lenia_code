import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import nltk

import torch
import hashlib
from typing import List, Union

import torch
import hashlib
from typing import Dict, Set
from pathlib import Path

from nltk.corpus import wordnet as wn




def fetch_nouns_and_adj_from_nltk(adj_path="adjectives.pickle", noun_path="nouns.pickle"):
    """
    this func is a one time call to download a database of nouns and adjectives to be stored as pickle files,
    the point is to avoid the database being changed in time since the nltk online version could be updated
    """
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('words', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception as e:
        raise Exception(f"Error downloading NLTK data: {str(e)}")

    adjectives: Set[str] = set()
    nouns: Set[str] = set()

    # Get adjectives
    for synset in list(wn.all_synsets(wn.ADJ)):
        for lemma in synset.lemmas():
            word = lemma.name().lower()
            if word.isalpha() and len(word) > 2:  # Filter out very short words and non-alphabetic
                adjectives.add(word)
    
    # Get nouns
    for synset in list(wn.all_synsets(wn.NOUN)):
        for lemma in synset.lemmas():
            word = lemma.name().lower()
            if word.isalpha() and len(word) > 2:  # Filter out very short words and non-alphabetic
                nouns.add(word)

    # Convert sets to sorted lists for deterministic behavior
    adjectives = sorted(list(adjectives))
    nouns = sorted(list(nouns))

    #save to files
    with open(adj_path, 'wb') as handle:
        pickle.dump(adjectives, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(noun_path, 'wb') as handle:
        pickle.dump(nouns, handle, protocol=pickle.HIGHEST_PROTOCOL)



def params_to_words(state_dict: Dict[str, torch.Tensor], num_words: int = 3, 
                    adj_path: str = "./utils/adjectives.pickle", 
                    noun_path: str = "./utils/nouns.pickle") -> str:
    """
    Convert neural network state dictionary into a deterministic sequence of words.
    Acts like a hash function - similar parameters will generate the same words.
    
    Args:
        state_dict: Model state dictionary loaded from torch.load()
        num_words: Number of words to generate
    
    Returns:
        String containing space-separated words
    """
    # Load adjectives and nouns
    with open(adj_path, 'rb') as handle:
        adjectives = pickle.load(handle)
    with open(noun_path, 'rb') as handle:
        nouns = pickle.load(handle)

    # Convert state dict to bytes for hashing
    param_bytes = b''
    for key in sorted(state_dict.keys()):  # Sort keys for deterministic ordering
        param_bytes += state_dict[key].cpu().numpy().tobytes()
    
    # Create deterministic seed from parameters
    hash_value = int(hashlib.sha256(param_bytes).hexdigest(), 16)
    
    # Generate words using the hash value
    words = []
    for i in range(num_words):
        seed = (hash_value + i * 12345) & 0xFFFFFFFF
        word_list = adjectives if i % 2 == 0 else nouns
        word_idx = seed % len(word_list)
        words.append(word_list[word_idx])
    
    return "_".join(words)


def get_params_name(params):
    new_params = params.copy()

    new_params["k_size"] = torch.tensor([params["k_size"]])
    new_params = dict([key, val] for key, val in new_params.items())
    name = params_to_words(new_params, num_words=2)
    return name



def restore_params_to_names(source_dir, target_dir, num_words, device="cuda"):
    """
    goes through all files in source_dir and stores the renamed files into target_dir
    """
    database={}
    for file in os.listdir(source_dir):
        print(file)
        dico = torch.load(os.path.join(source_dir,file), map_location=device, weights_only=True)
        params = dict([key, val] for key, val in dico.items() if key != "k_size")
        name = params_to_words(params, num_words)+".pt"
        out_file = os.path.join(target_dir,name)
        torch.save(dico, out_file)


#use once to generate files with word databases:
#fetch_nouns_and_adj_from_nltk()



#use case 1:
#source_dir = "../../demo_params/individual"
#device = "cuda"
#file = "intmu0.36_sigma0.07_0.69.pt"
#dico = torch.load(os.path.join(source_dir,file), map_location=device, weights_only=True)
#params = dict([key, val] for key, val in dico.items() if key != "k_size")
#name = params_to_words(params, num_words=2)+".pt"
#print(name)

#use case 2:
#source_dir = "../../demo_params/individual"
#target_dir = "../../demo_params/names"
#restore_params_to_names(source_dir, target_dir, num_words=2)


"""
the oldschool cute version
adjectives = [
        # Colors and Color Variations
        "crimson", "azure", "violet", "amber", "teal", "maroon", "cobalt", "scarlet", "indigo", 
        "emerald", "turquoise", "burgundy", "sage", "coral", "cerulean", "magenta", "ochre", "sienna",
        "chartreuse", "mauve", "auburn", "periwinkle", "vermillion", "fuchsia", "ivory", "russet",
        "aquamarine", "lavender", "olive", "beige", "khaki", "mahogany", "puce", "taupe", "sepia",
        "carmine", "viridian", "celadon", "aegean", "ultramarine", "byzantine", "amaranth", "zaffre",
        "pear", "thistle", "alabaster", "ebony", "garnet", "heliotrope", "jade", "malachite",
        "onyx", "ruby", "sapphire", "topaz", "verdigris", "wisteria", "xanadu", "zinc", "amethyst",
        
        # Weather and Nature
        "cloudy", "misty", "sunny", "stormy", "rainy", "windy", "foggy", "hazy", "humid", "arid",
        "tropical", "arctic", "alpine", "coastal", "verdant", "lush", "barren", "dewy", "frosty",
        "glacial", "temperate", "torrid", "wintry", "autumnal", "vernal", "ethereal", "sylvan",
        
        # Textures and Patterns
        "smooth", "rough", "silky", "coarse", "grainy", "sleek", "bumpy", "glossy", "matte", "polished",
        "textured", "velvety", "woolly", "zigzag", "spotted", "striped", "checkered", "dappled",
        "marbled", "speckled", "wrinkled", "crinkled", "woven", "pleated", "latticed", "ridged",
        
        # Qualities and Characteristics
        "gentle", "swift", "bright", "wild", "calm", "bold", "quiet", "fierce", "subtle", "sharp",
        "mellow", "vivid", "soft", "strong", "light", "deep", "warm", "cool", "fresh", "ancient",
        "ethereal", "mystical", "serene", "vibrant", "tranquil", "dynamic", "elegant", "graceful",
        "harmonious", "luminous", "radiant", "resplendent", "tenuous", "effervescent", "crystalline",
        "diaphanous", "gossamer", "iridescent", "lustrous", "mellifluous", "pellucid", "pristine",
        "quiescent", "resonant", "scintillating", "sublime", "ephemeral", "eternal", "infinite",
        "temporal", "celestial", "cosmic", "astral", "lunar", "solar", "stellar", "nebulous",
        
        # Seasonal and Time-Related
        "vernal", "estival", "autumnal", "hibernal", "diurnal", "nocturnal", "crepuscular",
        "matinal", "meridian", "vespertine", "perpetual", "fleeting", "enduring", "timeless",
        
        # Temperature and Climate
        "tepid", "torrid", "frigid", "gelid", "ambient", "balmy", "brisk", "chilly", "sultry",
        "temperate", "tropical", "wintry", "polar", "equatorial", "mediterranean", "oceanic",
        
        # Light and Shadow
        "brilliant", "dim", "gleaming", "glowing", "luminescent", "murky", "obscure", "opaque",
        "pellucid", "radiant", "shadowy", "shimmering", "translucent", "transparent", "umbral"
    ]
    
nouns = [
        # Fruits (Common and Exotic)
        "mango", "papaya", "kiwi", "peach", "plum", "pear", "fig", "apple", "cherry", "grape",
        "lemon", "lime", "orange", "apricot", "pomegranate", "nectarine", "tangerine", "persimmon",
        "guava", "lychee", "durian", "dragonfruit", "passionfruit", "mangosteen", "rambutan",
        "soursop", "custardapple", "breadfruit", "jackfruit", "quince", "mulberry", "boysenberry",
        "elderberry", "gooseberry", "kumquat", "longan", "miracle", "pawpaw", "salak", "tamarind",
        "ugli", "voavanga", "ximenia", "yangmei", "ziziphus", "calamansi", "carambola", "feijoa",
        
        # Vegetables and Herbs
        "carrot", "spinach", "celery", "pepper", "asparagus", "kale", "radish", "lettuce",
        "broccoli", "cabbage", "beet", "zucchini", "squash", "pea", "artichoke", "eggplant",
        "cucumber", "turnip", "parsnip", "leek", "arugula", "bamboo", "chicory", "daikon",
        "endive", "fennel", "galangal", "horseradish", "iceberg", "jicama", "kohlrabi",
        "lotus", "maca", "napa", "okra", "purslane", "quandong", "radicchio", "salsify",
        "taro", "ulluco", "wasabi", "yacon", "zebraplant", "amaranth", "borage", "cardoon",
        
        # Land Animals
        "tiger", "lion", "leopard", "jaguar", "cheetah", "lynx", "puma", "ocelot", "serval",
        "caracal", "wolf", "fox", "coyote", "jackal", "dhole", "deer", "elk", "moose",
        "antelope", "gazelle", "impala", "kudu", "oryx", "bison", "buffalo", "yak", "zebra",
        "giraffe", "okapi", "rhino", "hippo", "elephant", "tapir", "pangolin", "armadillo",
        "sloth", "anteater", "koala", "kangaroo", "wallaby", "wombat", "quokka", "numbat",
        "bandicoot", "possum", "quoll", "meerkat", "mongoose", "badger", "wolverine", "marten",
        
        # Birds
        "eagle", "hawk", "falcon", "osprey", "kite", "harrier", "vulture", "condor", "owl",
        "raven", "crow", "magpie", "jay", "cardinal", "finch", "sparrow", "warbler", "thrush",
        "blackbird", "starling", "swallow", "swift", "hummingbird", "kingfisher", "pelican",
        "flamingo", "stork", "crane", "heron", "egret", "ibis", "spoonbill", "penguin", "albatross",
        "petrel", "shearwater", "cormorant", "gannet", "frigatebird", "tropicbird", "pheasant",
        "quail", "partridge", "grouse", "ptarmigan", "peacock", "turkey", "guineafowl", "macaw",
        
        # Marine Life
        "dolphin", "whale", "orca", "porpoise", "seal", "sealion", "walrus", "manatee", "dugong",
        "shark", "ray", "skate", "salmon", "trout", "tuna", "marlin", "swordfish", "barracuda",
        "grouper", "snapper", "cod", "halibut", "flounder", "sole", "anchovy", "herring",
        "sardine", "mackerel", "seahorse", "octopus", "squid", "cuttlefish", "nautilus", "lobster",
        "crab", "shrimp", "prawn", "crayfish", "barnacle", "starfish", "urchin", "jellyfish",
        "anemone", "coral", "clam", "mussel", "oyster", "scallop", "abalone", "chiton", "limpet"
    ]

"""