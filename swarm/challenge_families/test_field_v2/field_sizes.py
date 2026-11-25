# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Which real Andorra areas hold which field sizes: the size table the field cutter draws an area from once the size
is drawn.

Each row is an area's folder name and one mark per size bin of field_cut.bin_edges(), a tenth of today's field
each, from half to one and a half times it: "1" when the area holds the bin, "." when it does not. An area holds a
bin when field_cut.holds passes at least 4 of 200 tries at the bin's largest size, with the cell layer where a person
may lie as the open ground. The table belongs to one area package (e493faa, 48 areas): rebuild it with
field_cut.size_table whenever the package or the cutter's rules change.

WOODED_ROWS is the same table for fields that reach into the trees (field_cut.WOODED), built the same way with that
recipe: 31 areas hold some wooded size, all the forests, villages and rivers and most towns and valleys, while the
scree and nearly every lake and high pasture hold too few trees by their open ground.

WEIGHTS is how an area is picked among those that hold a seed's size and kind: by chance in proportion to its weight.
An area that holds few sizes is asked for seldom, so an even pick gave ten areas half of all seeds and ten others
under one seed in a hundred each. The weights were fitted over the sizes and kinds the seeds ask for so that every
area that holds any size gets as near an equal share of the seeds as its sizes allow, between 1 and 256: with them
the most used area gets about 3 % of seeds and all but six get at least 1.8 %. An area the table does not list weighs 1.
Fit them again whenever a table or the share of wooded fields changes.
"""

from __future__ import annotations

from typing import Dict, Tuple

ROWS = {
    "01_estanys_de_l_angonella_comapedrosa": "1111111111",
    "02_estany_del_mig_tristaina": "111111111.",
    "03_estany_de_la_nou_madriu": "1111111111",
    "04_estany_de_baix_de_siscaro": "11111111..",
    "05_estany_de_querol_incles": "1111111111",
    "06_estany_de_montmalus": "1111111111",
    "07_l_obaga_hay_meadows": "1111111111",
    "08_el_rebollar": "11111111.1",
    "09_fontaneda_hay_meadows": "11111111..",
    "10_cami_ral_d_ordino_arans": "1111111111",
    "11_engolasters_hamlet_meadows": "11111.....",
    "12_sornas": "1111111111",
    "13_llorts": "1111111111",
    "14_pont_de_la_baladosa_incles": "1111111111",
    "15_bordes_de_rigoder": "111111....",
    "16_prats_valira_d_orient": "11........",
    "17_l_aldosa_de_la_massana": "11111111..",
    "18_canal_de_la_bova": "1111111111",
    "19_cami_de_l_alt_del_griu": "11........",
    "20_planell_dels_becos": "1111111...",
    "21_cortal_del_masover": "1111111111",
    "22_salador_de_l_andorrana": "..........",
    "23_pic_d_ortafa": "1111111111",
    "24_calm_de_claror": "1111111111",
    "25_planells_de_caraup": "11111111..",
    "26_port_de_rat": "11111111..",
    "27_la_tosa_de_les_mussoles": "1111111111",
    "28_pic_de_l_estanyo_grp1": "11........",
    "29_pla_de_les_pedres": "11111111..",
    "30_prat_dels_orris": "111111111.",
    "31_obac_de_soldeu": "111.......",
    "32_pic_dels_aspres_pal": "111.11....",
    "33_andorra_la_vella_test_field": "11111.....",
    "34_andorra_la_vella_valira_river": "1111111111",
    "35_sant_julia_de_loria_old_town": "11........",
    "36_sant_julia_de_loria_east_edge": "11111.....",
    "37_escaldes_caldea_park": "1111......",
    "38_escaldes_engordany_slope": "..........",
    "39_ordino_village_and_meadows": "1111111...",
    "40_ordino_river_meadows": "1111111111",
    "41_la_massana_valira_bank": "1111111111",
    "42_la_massana_village_fields": "1111111111",
    "43_encamp_centre": "1111111111",
    "44_encamp_stadium_and_river": "1111111111",
    "45_canillo_centre_and_river": "..........",
    "46_canillo_hillside_edge": "1111111111",
    "47_pas_de_la_casa_shopping_centre": "..........",
    "48_pas_de_la_casa_apartment_slope": "..........",
}


WOODED_ROWS = {
    "01_estanys_de_l_angonella_comapedrosa": "..........",
    "02_estany_del_mig_tristaina": "..........",
    "03_estany_de_la_nou_madriu": "1111111...",
    "04_estany_de_baix_de_siscaro": "..........",
    "05_estany_de_querol_incles": "..........",
    "06_estany_de_montmalus": "..........",
    "07_l_obaga_hay_meadows": "1111......",
    "08_el_rebollar": ".11.......",
    "09_fontaneda_hay_meadows": "1111.1....",
    "10_cami_ral_d_ordino_arans": "1111111111",
    "11_engolasters_hamlet_meadows": "..........",
    "12_sornas": "1111111111",
    "13_llorts": "1111111111",
    "14_pont_de_la_baladosa_incles": "1111111111",
    "15_bordes_de_rigoder": "111111....",
    "16_prats_valira_d_orient": ".1........",
    "17_l_aldosa_de_la_massana": "11111.....",
    "18_canal_de_la_bova": "1111111111",
    "19_cami_de_l_alt_del_griu": "11........",
    "20_planell_dels_becos": "11111.....",
    "21_cortal_del_masover": "1111111111",
    "22_salador_de_l_andorrana": "..........",
    "23_pic_d_ortafa": "..........",
    "24_calm_de_claror": "..........",
    "25_planells_de_caraup": "11........",
    "26_port_de_rat": "..........",
    "27_la_tosa_de_les_mussoles": "..........",
    "28_pic_de_l_estanyo_grp1": "..........",
    "29_pla_de_les_pedres": "11111.....",
    "30_prat_dels_orris": "..........",
    "31_obac_de_soldeu": "1.........",
    "32_pic_dels_aspres_pal": "..........",
    "33_andorra_la_vella_test_field": "111111....",
    "34_andorra_la_vella_valira_river": "111111111.",
    "35_sant_julia_de_loria_old_town": "11........",
    "36_sant_julia_de_loria_east_edge": "1.........",
    "37_escaldes_caldea_park": "111.......",
    "38_escaldes_engordany_slope": "..........",
    "39_ordino_village_and_meadows": "1.11......",
    "40_ordino_river_meadows": "1111111111",
    "41_la_massana_valira_bank": "1111111111",
    "42_la_massana_village_fields": "111111....",
    "43_encamp_centre": "111111111.",
    "44_encamp_stadium_and_river": "1111111111",
    "45_canillo_centre_and_river": "1.........",
    "46_canillo_hillside_edge": "1111......",
    "47_pas_de_la_casa_shopping_centre": "..........",
    "48_pas_de_la_casa_apartment_slope": "..........",
}


WEIGHTS = {
    "01_estanys_de_l_angonella_comapedrosa": 46,
    "02_estany_del_mig_tristaina": 65,
    "03_estany_de_la_nou_madriu": 3,
    "04_estany_de_baix_de_siscaro": 97,
    "05_estany_de_querol_incles": 46,
    "06_estany_de_montmalus": 46,
    "07_l_obaga_hay_meadows": 25,
    "08_el_rebollar": 44,
    "09_fontaneda_hay_meadows": 10,
    "10_cami_ral_d_ordino_arans": 1,
    "11_engolasters_hamlet_meadows": 256,
    "12_sornas": 1,
    "13_llorts": 1,
    "14_pont_de_la_baladosa_incles": 1,
    "15_bordes_de_rigoder": 8,
    "16_prats_valira_d_orient": 256,
    "17_l_aldosa_de_la_massana": 17,
    "18_canal_de_la_bova": 1,
    "19_cami_de_l_alt_del_griu": 256,
    "20_planell_dels_becos": 17,
    "21_cortal_del_masover": 1,
    "23_pic_d_ortafa": 46,
    "24_calm_de_claror": 46,
    "25_planells_de_caraup": 83,
    "26_port_de_rat": 97,
    "27_la_tosa_de_les_mussoles": 46,
    "28_pic_de_l_estanyo_grp1": 256,
    "29_pla_de_les_pedres": 17,
    "30_prat_dels_orris": 65,
    "31_obac_de_soldeu": 256,
    "32_pic_dels_aspres_pal": 249,
    "33_andorra_la_vella_test_field": 8,
    "34_andorra_la_vella_valira_river": 1,
    "35_sant_julia_de_loria_old_town": 256,
    "36_sant_julia_de_loria_east_edge": 256,
    "37_escaldes_caldea_park": 112,
    "39_ordino_village_and_meadows": 41,
    "40_ordino_river_meadows": 1,
    "41_la_massana_valira_bank": 1,
    "42_la_massana_village_fields": 7,
    "43_encamp_centre": 1,
    "44_encamp_stadium_and_river": 1,
    "45_canillo_centre_and_river": 256,
    "46_canillo_hillside_edge": 25,
}


def weights() -> Dict[str, int]:
    """Each area's weight in the pick among the areas that hold a seed's size and kind."""
    return dict(WEIGHTS)


def _read(rows: Dict[str, str]) -> Dict[str, Tuple[bool, ...]]:
    """A table's rows as the cutter reads them: for each area, whether it holds each size bin."""
    return {name: tuple(mark == "1" for mark in marks) for name, marks in rows.items()}


def table() -> Dict[str, Tuple[bool, ...]]:
    """The size table for open fields, as the cutter reads it."""
    return _read(ROWS)


def wooded_table() -> Dict[str, Tuple[bool, ...]]:
    """The size table for fields that reach into the trees, as the cutter reads it."""
    return _read(WOODED_ROWS)
