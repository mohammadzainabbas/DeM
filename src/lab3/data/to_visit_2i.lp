\* Paris_Visit_-_Max._number_of_sites_(must_visit_effiel_tower_and_catacombs_&_either_visit_Notre_Dame_Cathedral_or_Sainte_Chapelle_&_must_visit_the_Pompidou_Center_if_he_visits_Louvre) *\
Maximize
Max._number_of_sites: 5 AC + 3 AT + 4 BS + 4 CA + 5 CN + CP + 3 JT + 4 ML
 + 2 MO + 3 PC + SC + 5 TE + 2 TM
Subject To
Either_visit_TE_or_CA: CA + TE = 1
Max._duration: 1.5 AC + AT + 2 BS + 2 CA + 2 CN + 2.5 CP + 1.5 JT + 3 ML
 + 2 MO + 0.75 PC + 1.5 SC + 4.5 TE + 2 TM <= 12
Max._price: 9.5 AT + 8 BS + 10 CA + 5 CN + 10 CP + 12 ML + 11 MO + 8.5 SC
 + 15.5 TE + 15 TM <= 65
Must_visit_CA: CA = 1
Must_visit_CP_if_he_visits_ML: - CP + ML <= 0
Must_visit_TE: TE = 1
Binaries
AC
AT
BS
CA
CN
CP
JT
ML
MO
PC
SC
TE
TM
End
