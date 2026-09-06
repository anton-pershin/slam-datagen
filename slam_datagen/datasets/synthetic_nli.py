import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
import random
import networkx as nx
from pathlib import Path
from typing import List, Any
import json


@dataclass
class DatasetSample:
    id: int # уникальный идентификатор примера
    premise: str # текст посылки (может состоять из нескольких предложений)
    hypothesis: str # текст гипотезы (одно предложение)
    gold_label: str # одна из меток entailment | neutral | contradiction
    split: str # train | eval | (опционально) test
    hop_difficulty: int # целое число из множества {0,1,2}
    distractor_difficulty: int # целое число из множества {1,2,3}
    background_knowledge: str # explicit | implicit
    world_seed: str # как сделать!? сид, использованный для генерации скрытого мира (для воспроизводимости) 
    templates_version: str # что это?
    ood_tag: str # режим разбиения iid | entity_heldout | taxonomy_heldout | relation_pattern_heldout


def build_synthetic_nli_dataset(cfg: DictConfig) -> list[DatasetSample]:
    W = HiddenWord(n_entities=cfg.n_entities)
    W.sample_hiden_world()
    samples = []
    
    distractor_difficulty = [item.value for item in cfg.distractor_difficulty_distribution]
    distractor_weights = [item.weight for item in cfg.distractor_difficulty_distribution]
    hop_difficulty = [item.value for item in cfg.hop_difficulty_distribution]
    hop_weights = [item.weight for item in cfg.hop_difficulty_distribution]

    # idk how to not hardcode that
    chosen_distractors = random.choices(distractor_difficulty, weights=distractor_weights, k=cfg.n_train)
    choosen_hop = random.choices(hop_difficulty, weights=hop_weights, k=cfg.n_train)
    for i in range(cfg.n_train):
        if choosen_hop[i] == 0:
            samples.append(W.generate_zero_hop_sample(distractor_difficulty=chosen_distractors[i], split="train"))
        elif choosen_hop[i] == 1:
            samples.append(W.generate_one_hop_sample(distractor_difficulty=chosen_distractors[i], background_knowledge="explicit", split="train"))
        elif choosen_hop[i] == 2:
            samples.append(W.generate_two_hop_sample(distractor_difficulty=chosen_distractors[i], background_knowledge="explicit", split="train"))

    chosen_distractors = random.choices(distractor_difficulty, weights=distractor_weights, k=cfg.n_eval)
    choosen_hop = random.choices(hop_difficulty, weights=hop_weights, k=cfg.n_eval)
    for i in range(cfg.n_eval):
        if choosen_hop[i] == 0:
            samples.append(W.generate_zero_hop_sample(distractor_difficulty=chosen_distractors[i], split="eval"))
        elif choosen_hop[i] == 1:
            samples.append(W.generate_one_hop_sample(distractor_difficulty=chosen_distractors[i], background_knowledge="explicit", split="eval"))
        elif choosen_hop[i] == 2:
            samples.append(W.generate_two_hop_sample(distractor_difficulty=chosen_distractors[i], background_knowledge="explicit", split="eval"))
    
    return samples

def write_synthetic_nli_dataset(
    samples: list[DatasetSample],
    output_file: str | Path,
) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(_serialize_sample(sample)) + "\n")

    return output_path

def _serialize_sample(sample: DatasetSample) -> dict[str, Any]:
    return {
        "id": sample.id,
        "premise": sample.premise,
        "hypothesis": sample.hypothesis,
        "gold_label": sample.gold_label,
        "split": sample.split,
        "hop_difficulty": sample.hop_difficulty,
        "distractor_difficulty": sample.distractor_difficulty,
        "background_knowledge": sample.background_knowledge,
        "world_seed": sample.world_seed,
        "templates_version": sample.templates_version,
        "ood_tag": sample.ood_tag,
    }

class HiddenWord():
    def __init__(self, n_entities, templates_version=0, word_seed=11, ood_tag="iid"):
        self.last_used_id = 0
        self.n_entities = n_entities
        self.word_seed = word_seed
        self.ood_tag = ood_tag # todo realise different tags
        self.templates_version = templates_version

        random.seed(self.word_seed)

    def sample_hiden_world(self):
        colors = [
            "red", "green", "blue", "yellow", "orange",
            "purple", "pink", "brown", "black", "white",
            "gray", "cyan", "magenta", "lime", "maroon",
            "navy", "olive", "teal", "silver", "gold",
            "coral", "indigo", "violet", "turquoise", "beige"
        ]

        locations = [
            "park", "forest", "beach", "mountain", "river",
            "lake", "ocean", "desert", "island", "valley",
            "cave", "waterfall", "volcano", "glacier", "jungle",
            "savanna", "tundra", "swamp", "canyon", "prairie",
            "reef", "coast", "plateau", "meadow", "hill"
        ]

        shapes = [
            "circle", "square", "triangle", "rectangle", "oval",
            "pentagon", "hexagon", "octagon", "diamond", "star",
            "crescent", "heart", "arrow", "cross", "spiral",
            "cylinder", "cube", "sphere", "cone", "pyramid",
            "torus", "prism", "parallelogram", "trapezoid", "ellipse"
        ]

        animals = [
            "dog", "cat", "elephant", "lion", "tiger",
            "bear", "zebra", "giraffe", "monkey", "dolphin",
            "whale", "shark", "eagle", "snake", "frog",
            "rabbit", "horse", "cow", "pig", "sheep",
            "wolf", "fox", "red panda", "kangaroo", "penguin"
        ]

        # ? made dynamyc range of types
        types = [f"t{i}" for i in range(1000)]
        chosen_types = random.sample(types, 500)
 
        entities = [f"e{i}" for i in range(self.n_entities * 2)]
        chosen_entities = random.sample(entities, self.n_entities)

        self.zero_hop_dict = dict()
        self.one_hop_dict = dict()
        self.one_hop_DAG = nx.DiGraph()
        self.two_hop_DAG = nx.DiGraph()

        self.one_hop_DAG.add_nodes_from(chosen_types)
        self.two_hop_DAG.add_nodes_from(chosen_entities)

        self.add_edges_to_DAG(self.one_hop_DAG, list(self.one_hop_DAG.nodes()))
        self.add_edges_to_DAG(self.two_hop_DAG, list(self.two_hop_DAG.nodes()))

        leaf_types = [t for t in self.one_hop_DAG.nodes() if self.one_hop_DAG.in_degree(t) == 0]
        self.leaf_types = leaf_types
        self.add_true_label_to_all_entities(self.zero_hop_dict,
                                            random.sample(colors + locations + shapes + animals, 50),
                                            chosen_entities)
        self.add_true_label_to_all_entities(self.one_hop_dict,
                                            leaf_types,
                                            chosen_entities)

    def add_true_label_to_all_entities(self, d, labels: List[str], entities: List[str]):
        for entity in entities:
            d[entity] = random.choice(labels)

    def add_edges_to_DAG(self, DAG: nx.DiGraph, nodes: List[str], p=0.35):
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if random.random() < p:
                    DAG.add_edge(nodes[i], nodes[j])

    def generate_zero_hop_sample(self, distractor_difficulty, split) -> DatasetSample:
        print("zero")
        gold_label = random.choice(["entailment", "neutral", "contradiction"])
        entity = random.choice(list(self.zero_hop_dict.keys()))
        chosen_entities = {entity}
        premise = []

        if gold_label == "entailment":
            premise.append(f"{entity} is {self.zero_hop_dict[entity]}")
            hypothesis = premise[0]
        elif gold_label == "neutral":
            premise.append(f"{entity} is {self.zero_hop_dict[entity]}")
            entity = random.choice(list(self.zero_hop_dict.keys()))
            while entity in chosen_entities:
                entity = random.choice(list(self.zero_hop_dict.keys()))
            hypothesis = f"{entity} is {self.zero_hop_dict[entity]}"
            chosen_entities.add(entity)
        elif gold_label == "contradiction":
            premise.append(f"{entity} is {self.zero_hop_dict[entity]}")
            false_label = random.choice(list(self.zero_hop_dict.values()))
            while false_label == self.zero_hop_dict[entity]:
                false_label = random.choice(list(self.zero_hop_dict.values()))
            hypothesis = f"{entity} is {false_label}"
            chosen_entities = {entity}
            
        # ? надо ли чтобы не относящиеся к делу факты были реальными (сейчас реализовано не так)
        while len(chosen_entities) != distractor_difficulty + 1:
            entity = random.choice(list(self.zero_hop_dict.keys()))
            if entity not in chosen_entities:
                chosen_entities.add(entity)
                premise.append(f"{entity} is {random.choice(list(self.zero_hop_dict.values()))}")
        
        self.last_used_id += 1
        random.shuffle(premise)
        sample = DatasetSample(self.last_used_id, premise, hypothesis, gold_label, split, 0, distractor_difficulty, "explicit", self.word_seed, self.templates_version, self.ood_tag)

        return sample

    def generate_one_hop_sample(self, distractor_difficulty, background_knowledge, split) -> DatasetSample:
        print("one")
        gold_label = random.choice(["entailment", "neutral", "contradiction"])
        entity = random.choice(list(self.one_hop_dict.keys()))
        child_type = self.one_hop_dict[entity]
        parents = list(self.one_hop_DAG.successors(child_type))
        while not parents:
            entity = random.choice(list(self.one_hop_dict.keys()))
            child_type = self.one_hop_dict[entity]
            parents = list(self.one_hop_DAG.successors(child_type))
        parent_type = random.choice(parents)
        
        chosen_entities = {entity}
        premise = [f"{entity} is a {child_type}"]

        if background_knowledge == "explicit":
            premise.append(f"{child_type} is a subtype of {parent_type}")

        if gold_label == "entailment":
            hypothesis = f"{entity} is a {parent_type}"
        elif gold_label == "neutral":
            other_entity = random.choice(list(self.one_hop_dict.keys()))
            while other_entity == entity:
                other_entity = random.choice(list(self.one_hop_dict.keys()))
            other_type = random.choice(list(self.one_hop_DAG.nodes()))
            hypothesis = f"{other_entity} is a {other_type}"
            chosen_entities.add(other_entity)
        # ? is that ok
        elif gold_label == "contradiction":
            other_leaf = random.choice([t for t in self.leaf_types if t != child_type])
            hypothesis = f"{entity} is a {other_leaf}"

        while len(chosen_entities) != distractor_difficulty + 1:
            other_entity = random.choice(list(self.one_hop_dict.keys()))
            if other_entity not in chosen_entities:
                chosen_entities.add(other_entity)
                other_type = self.one_hop_dict[other_entity]
                premise.append(f"{other_entity} is a {other_type}")

        
        self.last_used_id += 1
        random.shuffle(premise)
        sample = DatasetSample(self.last_used_id, premise, hypothesis, gold_label, split, 1, distractor_difficulty, background_knowledge, self.word_seed, self.templates_version, self.ood_tag)

        return sample
    
    def generate_two_hop_sample(self, distractor_difficulty, background_knowledge, split) -> DatasetSample:
        print("two")
        gold_label = random.choice(["entailment", "neutral", "contradiction"])
        use_taxonomy = random.random() < 0.5

        if gold_label == "entailment":
            if use_taxonomy:
                entity = None
                leaf_type = None
                parent_type = None
                grandparent_type = None
                for _ in range(100):
                    e = random.choice(list(self.one_hop_dict.keys()))
                    leaf = self.one_hop_dict[e]
                    parents = list(self.one_hop_DAG.successors(leaf))
                    if not parents:
                        continue
                    p = random.choice(parents)
                    grandparents = list(self.one_hop_DAG.successors(p))
                    if grandparents:
                        entity = e
                        leaf_type = leaf
                        parent_type = p
                        grandparent_type = random.choice(grandparents)
                        break
                if entity is not None:
                    premise = [f"{entity} is a {leaf_type}"]
                    if background_knowledge == "explicit":
                        premise.append(f"{leaf_type} is a subtype of {parent_type}")
                        premise.append(f"{parent_type} is a subtype of {grandparent_type}")
                    hypothesis = f"{entity} is a {grandparent_type}"
                    chosen_entities = {entity}
                    chosen_types = {leaf_type, parent_type, grandparent_type}
                else:
                    use_taxonomy = False

            else:
                a = b = c = None
                for _ in range(100):
                    b_candidate = random.choice(list(self.two_hop_DAG.nodes()))
                    c_candidates = list(self.two_hop_DAG.successors(b_candidate))
                    if not c_candidates:
                        continue
                    a_candidates = list(self.two_hop_DAG.predecessors(b_candidate))
                    if a_candidates:
                        a = random.choice(a_candidates)
                        b = b_candidate
                        c = random.choice(c_candidates)
                        break
                premise = [f"{a} is left of {b}", f"{b} is left of {c}"]
                if background_knowledge == "explicit":
                    premise.append("Rule: left_of is transitive and antisymmetric")
                hypothesis = f"{a} is left of {c}"
                chosen_entities = {a, b, c}
        elif gold_label == "neutral":
            if use_taxonomy:
                entity = None
                leaf_type = None
                parent_type = None
                grandparent_type = None
                for _ in range(100):
                    e = random.choice(list(self.one_hop_dict.keys()))
                    leaf = self.one_hop_dict[e]
                    parents = list(self.one_hop_DAG.successors(leaf))
                    if not parents:
                        continue
                    p = random.choice(parents)
                    grandparents = list(self.one_hop_DAG.successors(p))
                    if grandparents:
                        entity = e
                        leaf_type = leaf
                        parent_type = p
                        grandparent_type = random.choice(grandparents)
                        break
                if entity is not None:
                    premise = [f"{entity} is a {leaf_type}"]
                    if background_knowledge == "explicit":
                        premise.append(f"{leaf_type} is a subtype of {parent_type}")
                        premise.append(f"{parent_type} is a subtype of {grandparent_type}")
                    other_entity = random.choice(list(self.one_hop_dict.keys()))
                    while other_entity == entity:
                        other_entity = random.choice(list(self.one_hop_dict.keys()))
                    other_type = random.choice(list(self.one_hop_DAG.nodes()))
                    hypothesis = f"{other_entity} is a {other_type}"
                    chosen_entities = {entity, other_entity}
                    chosen_types = {leaf_type, parent_type, grandparent_type, other_type}
                else:
                    use_taxonomy = False
            else:
                edges = list(self.two_hop_DAG.edges())
                (a1, b1), (a2, b2) = random.sample(edges, 2)
                premise = [f"{a1} is left of {b1}", f"{a2} is left of {b2}"]
                if background_knowledge == "explicit":
                    premise.append("Rule: left_of is transitive")
                hypothesis = f"{a1} is left of {a2}" 
                chosen_entities = {a1, b1, a2, b2}
        elif gold_label == "contradiction":
            if use_taxonomy:
                entity = None
                leaf_type = None
                parent_type = None
                grandparent_type = None
                for _ in range(100):
                    e = random.choice(list(self.one_hop_dict.keys()))
                    leaf = self.one_hop_dict[e]
                    parents = list(self.one_hop_DAG.successors(leaf))
                    if not parents:
                        continue
                    p = random.choice(parents)
                    grandparents = list(self.one_hop_DAG.successors(p))
                    if grandparents:
                        entity = e
                        leaf_type = leaf
                        parent_type = p
                        grandparent_type = random.choice(grandparents)
                        break
                if entity is not None:
                    leaf_types = [t for t in self.one_hop_DAG.nodes() if self.one_hop_DAG.in_degree(t) == 0]
                    other_leaf = random.choice([t for t in leaf_types if t != leaf_type])
                    premise = [f"{entity} is a {leaf_type}"]
                    if background_knowledge == "explicit":
                        premise.append(f"{leaf_type} is a subtype of {parent_type}")
                        premise.append(f"{parent_type} is a subtype of {grandparent_type}")
                    hypothesis = f"{entity} is a {other_leaf}"
                    chosen_entities = {entity}
                    chosen_types = {leaf_type, parent_type, grandparent_type, other_leaf}
                else:
                    use_taxonomy = False

            else:
                a = b = c = None
                for _ in range(100):
                    b_candidate = random.choice(list(self.two_hop_DAG.nodes()))
                    c_candidates = list(self.two_hop_DAG.successors(b_candidate))
                    if not c_candidates:
                        continue
                    a_candidates = list(self.two_hop_DAG.predecessors(b_candidate))
                    if a_candidates:
                        a = random.choice(a_candidates)
                        b = b_candidate
                        c = random.choice(c_candidates)
                        break
                premise = [f"{a} is left of {b}", f"{b} is left of {c}"]
                if background_knowledge == "explicit":
                    premise.append("Rule: left_of is transitive and antisymmetric")
                hypothesis = f"{c} is left of {a}"
                chosen_entities = {a, b, c}

        added_distractors = 0
        while added_distractors < distractor_difficulty:
            if use_taxonomy:
                other_entity = random.choice(list(self.one_hop_dict.keys()))
                if other_entity not in chosen_entities:
                    chosen_entities.add(other_entity)
                    other_type = self.one_hop_dict[other_entity]
                    premise.append(f"{other_entity} is a {other_type}")
                    added_distractors += 1
            else:
                other_a = random.choice(list(self.two_hop_DAG.nodes()))
                other_b_candidates = list(self.two_hop_DAG.successors(other_a))
                if not other_b_candidates:
                    continue
                other_b = random.choice(other_b_candidates)
                if other_a not in chosen_entities:
                    chosen_entities.add(other_a)
                if other_b not in chosen_entities:
                    chosen_entities.add(other_b)
                premise.append(f"{other_a} is left of {other_b}")
                added_distractors += 1


        self.last_used_id += 1
        random.shuffle(premise)
        sample = DatasetSample(self.last_used_id, premise, hypothesis, gold_label, split, 2, distractor_difficulty, background_knowledge, self.word_seed, self.templates_version, self.ood_tag)

        return sample

if __name__ == "__main__":
    W = HiddenWord(1)
    W.sample_hiden_world()