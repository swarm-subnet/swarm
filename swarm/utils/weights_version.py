import re
import random


def version_to_tuple(version):
    return tuple(map(int, version.split('.')))


def is_valid_version_format(version):
    return bool(re.match(r'^\d+\.\d+\.\d+$', version))


def is_version_in_range(version, version1, version2):
    if not is_valid_version_format(version):
        return False

    v = version_to_tuple(version)
    v1 = version_to_tuple(version1)
    v2 = version_to_tuple(version2)

    if v1 > v2:
        v1, v2 = v2, v1

    return v1 <= v <= v2


def tuple_to_version(version_tuple):
    return '.'.join(map(str, version_tuple))


def generate_random_version(version1, version2):
    v1 = version_to_tuple(version1)
    v2 = version_to_tuple(version2)

    if v1 > v2:
        v1, v2 = v2, v1

    def random_version_near(v):
        return tuple(
            max(v[i] + random.choice([-1, 1]), 0)
            if random.random() > 0.5 else v[i]
            for i in range(len(v))
        )

    def is_in_range(v):
        return v1 <= v <= v2

    while True:
        random_near_v1 = random_version_near(v1)
        if not is_in_range(random_near_v1):
            return tuple_to_version(random_near_v1)

        random_near_v2 = random_version_near(v2)
        if not is_in_range(random_near_v2):
            return tuple_to_version(random_near_v2)
