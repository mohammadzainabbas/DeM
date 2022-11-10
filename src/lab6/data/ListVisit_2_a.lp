\* Paris_Visit_-_Max._number_of_sites_(within_1_km_radius) *\
Maximize
Max._number_of_sites: 5 AC + 3 AT + 4 BS + 4 CA + 5 CN + CP + 3 JT + 4 ML
 + 2 MO + 3 PC + SC + 5 TE + 2 TM
Subject To
Distance_between_AC_and_AT: AC - AT = 0
Distance_between_AT_and_AC: - AC + AT = 0
Distance_between_CN_and_CP: CN - CP = 0
Distance_between_CN_and_SC: CN - SC = 0
Distance_between_CP_and_CN: - CN + CP = 0
Distance_between_CP_and_SC: CP - SC = 0
Distance_between_JT_and_MO: JT - MO = 0
Distance_between_JT_and_PC: JT - PC = 0
Distance_between_ML_and_SC: ML - SC = 0
Distance_between_MO_and_JT: - JT + MO = 0
Distance_between_MO_and_PC: MO - PC = 0
Distance_between_PC_and_JT: - JT + PC = 0
Distance_between_PC_and_MO: - MO + PC = 0
Distance_between_SC_and_CN: - CN + SC = 0
Distance_between_SC_and_CP: - CP + SC = 0
Distance_between_SC_and_ML: - ML + SC = 0
Max._duration: 1.5 AC + AT + 2 BS + 2 CA + 2 CN + 2.5 CP + 1.5 JT + 3 ML
 + 2 MO + 0.75 PC + 1.5 SC + 4.5 TE + 2 TM <= 12
Max._price: 9.5 AT + 8 BS + 10 CA + 5 CN + 10 CP + 12 ML + 11 MO + 8.5 SC
 + 15.5 TE + 15 TM <= 65
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
