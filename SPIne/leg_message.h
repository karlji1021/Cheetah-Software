#ifndef _leg_message
#define _leg_message

#include <stdint.h>

// 60 bytes
// 30 16-bit words
struct spi_data_t
{
    float q_abad[2];
    float q_hip[2];
    float q_knee[2];
    float qd_abad[2];
    float qd_hip[2];
    float qd_knee[2];
    int32_t flags[2];
    int32_t checksum;
};

// 132 bytes
// 66 16-bit words
struct spi_command_t
{
    float q_des_abad[2];
    float q_des_hip[2];
    float q_des_knee[2];
    float qd_des_abad[2];
    float qd_des_hip[2];
    float qd_des_knee[2];
    float kp_abad[2];
    float kp_hip[2];
    float kp_knee[2];
    float kd_abad[2];
    float kd_hip[2];
    float kd_knee[2];
    float tau_abad_ff[2];
    float tau_hip_ff[2];
    float tau_knee_ff[2];
    int32_t flags[2];
    int32_t checksum;
};



struct joint_control{
    float p_des, v_des, kp, kd, t_ff;
    };
    
struct joint_state{
    float p, v, t;
    };
    
struct leg_state{
    joint_state a, h, k;
    };
struct leg_control{
    joint_control a, h, k;
    }
    ;
#endif