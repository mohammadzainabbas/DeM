/* Objective function */
min: s_a+s_b+s_d+s_i+s_l;

/* Variable bounds */

/* Global utility of each alternative*/

U_A=u_p_17+u_c_17;
U_B=u_p_16+u_c_18;
U_C=u_p_10+u_c_17;
U_D=u_p_11+u_c_17;
U_E=u_p_12+u_c_12;
U_F=u_p_13+u_c_11;
U_G=u_p_14+u_c_8;
U_H=u_p_11+u_c_9;
U_I=u_p_13+u_c_8;
U_K=u_p_14+u_c_5;
U_L=u_p_11+u_c_6;
U_M=u_p_12+u_c_4;

/* Computation of the error of each alternative of the reference subset*/
V_A=U_A+s_a;
V_B=U_B+s_b;
V_D=U_D+s_d;
V_I=U_I+s_i;
V_L=U_L+s_l;

/* DM's preferences*/

delta=0.1;
V_A > V_B + delta;
V_B > V_D + delta;
V_D > V_I + delta;
V_I > V_L + delta;

/* Monotonicity constraints of criterion performance*/

delta_bis=2;
u_p_17 > u_p_16 + delta_bis;
u_p_16 > u_p_14 + delta_bis;
u_p_14 > u_p_13 + delta_bis;
u_p_13 > u_p_12 + delta_bis;
u_p_12 > u_p_11+ delta_bis;
u_p_11 > u_p_10+ delta_bis;

/* Monotonicity constraints of criterion composition*/
u_c_19 > u_c_18 + delta_bis;
u_c_18 > u_c_17 + delta_bis;
u_c_17 > u_c_12 + delta_bis;
u_c_12 > u_c_11 + delta_bis;
u_c_11 > u_c_10 + delta_bis;
u_c_10 > u_c_9 + delta_bis;
u_c_9 > u_c_8 + delta_bis;
u_c_8 > u_c_6 + delta_bis;
u_c_6 > u_c_5 + delta_bis;
u_c_5 > u_c_4 + delta_bis;
