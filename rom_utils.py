import numpy as np

def compute_penalty_coefficients(mapping_list, hm_avg_list, h_th, E, nu, penalty_coefficient=1000.):
    alpha_d_list = []
    alpha_r_list = []

    for i in range(len(mapping_list)):
        s_ind0, s_ind1 = mapping_list[i]

        h_th0 = h_th[s_ind0]
        h_th1 = h_th[s_ind1]

        max_Aij0 = E*h_th0/(1-nu**2)
        max_Aij1 = E*h_th1/(1-nu**2)
        max_Dij0 = E*(h_th0**3)/(12*(1-nu**2))
        max_Dij1 = E*(h_th1**3)/(12*(1-nu**2))

        alpha_d = penalty_coefficient/hm_avg_list[i]*np.min(max_Aij0, max_Aij1)
        alpha_r = penalty_coefficient/hm_avg_list[i]*np.min(max_Dij0, max_Dij1)
        # elif method == 'maximum':
        #     alpha_d = Constant(self.penalty_coefficient)\
        #                 /self.hm_avg_list[i]*max_value(max_Aij0, max_Aij1)
        #     alpha_r = Constant(self.penalty_coefficient)\
        #                 /self.hm_avg_list[i]*max_value(max_Dij0, max_Dij1)
        # elif method == 'average':
        #     alpha_d = Constant(self.penalty_coefficient)\
        #                 /self.hm_avg_list[i]*(max_Aij0+max_Aij1)*0.5
        #     alpha_r = Constant(self.penalty_coefficient)\
        #                 /self.hm_avg_list[i]*(max_Dij0+max_Dij1)*0.5
        # else:
        #     raise TypeError("Method:", method, "is not supported.")
        alpha_d_list += [alpha_d,]
        alpha_r_list += [alpha_r,]
    return alpha_d_list, alpha_r_list