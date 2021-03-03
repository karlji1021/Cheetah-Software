#include "JPos_Controller.hpp"
#include <random>


void karl_Controller::initializeController() {
  _loadPath = "/home/user/raisim_workspace/Cheetah-Software/actor_model/actor_1700.pt";
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    _actor = torch::jit::load(_loadPath);
  }
  catch (const c10::Error& e) {
    std::cerr << "!!!!Error loading the model!!!!\n";
  }
  _obsDim = 34;

  _obsMean.setZero(_obsDim);
  _obsVar.setZero(_obsDim);
  std::string in_line;
  std::ifstream obsMean_file("/home/user/raisim_workspace/Cheetah-Software/actor_model/mean1700.csv");
  std::ifstream obsVariance_file("/home/user/raisim_workspace/Cheetah-Software/actor_model/var1700.csv");
  if(obsMean_file.is_open()) {
    for(int i = 0; i < _obsMean.size(); i++){
      std::getline(obsMean_file, in_line);
      _obsMean(i) = std::stod(in_line);
    }
  }
  if(obsVariance_file.is_open()) {
    for(int i = 0; i < _obsVar.size(); i++){
      std::getline(obsVariance_file, in_line);
      _obsVar(i) = std::stod(in_line);
    }
  }
  obsMean_file.close();
  obsVariance_file.close();

//  RotMat<float> a;
//  Quat<float> q;
//  q[0] = 0.5, q[1] = 0.5, q[2] = 0.5, q[3] = 0.5;
//  a = ori::quaternionToRotationMatrix(q);
//  std::cout << "a: " << a.transpose().row(2) << std::endl;

}

void karl_Controller::runController(){
  Eigen::Matrix<float, 12, 1> qInitVec;
  Vec3<float> qInitVec1, qInitVec2, qInitVec3, qInitVec4;
  Mat3<float> kpMat;
  Mat3<float> kdMat;
  //kpMat << 20, 0, 0, 0, 20, 0, 0, 0, 20;
  //kdMat << 2.1, 0, 0, 0, 2.1, 0, 0, 0, 2.1;
  qInitVec1 << userParameters.q_init1[0], userParameters.q_init1[1], userParameters.q_init1[2];
  qInitVec2 << userParameters.q_init2[0], userParameters.q_init2[1], userParameters.q_init2[2];
  qInitVec3 << userParameters.q_init3[0], userParameters.q_init3[1], userParameters.q_init3[2];
  qInitVec4 << userParameters.q_init4[0], userParameters.q_init4[1], userParameters.q_init4[2];
  qInitVec << qInitVec1, qInitVec2, qInitVec3, qInitVec4;
  kpMat << userParameters.kp[0], 0, 0, 0,  userParameters.kp[1], 0, 0, 0,  userParameters.kp[2];
  kdMat <<  userParameters.kd[0], 0, 0, 0, userParameters.kd[1], 0, 0, 0, userParameters.kd[2];

//  std::cout << "orientation: " << _stateEstimate->orientation << std::endl;

  /// TODO: check each variable is appropriate.
  _bodyHeight = _stateEstimate->position(2);  // body height 1
  _bodyOri << _stateEstimate->rBody.transpose().row(2).transpose(); // body orientation 3
  _jointQ << _legController->datas[0].q, _legController->datas[1].q, _legController->datas[2].q, _legController->datas[3].q;  // joint angles 3 3 3 3 = 12
  _bodyVel << _stateEstimate->vWorld;  // velocity 3
  _bodyAngularVel << _stateEstimate->omegaWorld;  // angular velocity 3
  _jointQd << _legController->datas[0].qd, _legController->datas[1].qd, _legController->datas[2].qd, _legController->datas[3].qd;  // joint velocity 3 3 3 3 = 12

  _obs.setZero(_obsDim);
  _obs << _bodyHeight,
    _bodyOri,
    _jointQ,
    _bodyVel,
    _bodyAngularVel,
    _jointQd;

  static int iter(0);
  ++iter;

//  std::cout << "obs: " << _obs << std::endl;
  ///
  if(iter%1000 == 0) {
    std::cout << "Test 1: Observation before normalization." << std::endl;
    std::cout << _obs << std::endl;
  }
  ///

  for(int i = 0; i < _obs.size(); i++) {
    _obs(i) = (_obs(i) - _obsMean(i)) / std::sqrt(_obsVar(i) + 1e-8);
    if(_obs(i) > 10) _obs(i) = 10.0;
    if(_obs(i) < -10) _obs(i) = -10.0;
  }

  ///
//  std::cout << "Test 2-1: Observation after normalization, but without being converted to tensor." << std::endl;
//  std::cout << _obs << std::endl;
  ///

  _input.push_back(torch::jit::IValue(torch::from_blob(_obs.data(), {_obs.cols(),_obs.rows()}).clone()));
//  std::cout << "observation: " << torch::from_blob(_obs.data(), {_obs.cols(),_obs.rows()}) << std::endl;
  torch::Tensor action_tensor = _actor.forward(_input).toTensor();  // action of Tensor type
  float* action_f = action_tensor.data_ptr<float>();  // action of float pointer type
  Eigen::Map<Eigen::MatrixXf> action_eigen(action_f, action_tensor.size(0), action_tensor.size(1));  // action of MatrixXf type
  _input.pop_back();

  ///
//  std::cout << "Test 3: Actor must have the same output for the same input." << std::endl;
//  Eigen::VectorXf obs_test;
//  obs_test.setZero(_obsDim);
//  _input.push_back(torch::jit::IValue(torch::from_blob(obs_test.data(), {obs_test.cols(),obs_test.rows()}).clone()));
//  torch::Tensor action_tensor_test = _actor.forward(_input).toTensor();  // action of Tensor type
//  float* action_f_test = action_tensor_test.data_ptr<float>();  // action of float pointer type
//  Eigen::Map<Eigen::MatrixXf> action_eigen_test(action_f_test, action_tensor_test.size(0), action_tensor_test.size(1));  // action of MatrixXf type
//  _input.pop_back();
//  std::cout << "test input: " << obs_test << std::endl;
//  std::cout << "action: " << action_eigen_test << std::endl;
//  std::cout << "action dimension: " << action_eigen_test.size() << std::endl;
//
//  std::cout << "Test 4: action scaling." << std::endl;
//  std::cout << "For the same input action, normalized action output must be the same." << std::endl;
  ///

  // action scaling
  Eigen::VectorXd q_init;
  Eigen::VectorXd pTarget_12;
  q_init.setZero(12);
  pTarget_12.setZero(12);
//  q_init << -0.726685, -0.947298, 2.7, 0.726636, -0.947339, 2.7, -0.727, -0.94654, 2.65542, 0.727415, -0.946541, 2.65542;
  q_init << 0, -0.7854, 1.8326, 0, -0.7854, 1.8326, 0, -0.7854, 1.8326, 0, -0.7854, 1.8326;
  pTarget_12 = action_eigen.row(0).cast<double>();
  pTarget_12 *= 0.3;
  pTarget_12 += q_init;
//  std::cout << "input: " << action_tensor << std::endl;
//  std::cout << "output: " << pTarget_12 << std::endl;

  ///
//  std::cout << "Test 5: P Gain & D Gain." << std::endl;
//  std::cout << "Kp: " << kpMat << std::endl;
//  std::cout << "Kd: " << kdMat << std::endl;
  ///

  _legController->_maxTorque = 18;
  _legController->_legsEnabled = true;

  if(iter < 10){
    for(int leg(0); leg<4; ++leg){
      for(int jidx(0); jidx<3; ++jidx){
        _jpos_ini[3*leg+jidx] = _legController->datas[leg].q[jidx];
      }
    }
    return;
  } else if(iter < 1000) {
    for(int leg(0); leg<4; ++leg){
      for(int jidx(0); jidx<3; ++jidx){
        _legController->commands[leg].qDes[jidx] = _jpos_ini[3*leg+jidx] + iter/999.*(qInitVec(leg*3 + jidx)-_jpos_ini[3*leg+jidx]);
        _legController->commands[leg].qdDes[jidx] = 0.;
        _legController->commands[leg].tauFeedForward[jidx] = userParameters.tau_ff;
      }
      _legController->commands[leg].kpJoint = kpMat*10;
      _legController->commands[leg].kdJoint = kdMat*10;

//      std::cout << "Joint position: " << _legController->datas[leg].q << std::endl;
    }

    return;
  }

//  if(iter % 1000 == 0) {
//    std::cout << "observation after normalization: " << std::endl;
//    std::cout << "obs: " << _obs << std::endl;
//    std::cout << "joint target after scaling: " << std::endl;
//    std::cout << "pTarget12: " << pTarget_12 << std::endl;
//  }



  if(userParameters.calibrate > 0.4) {
    _legController->_calibrateEncoders = userParameters.calibrate;
  } else {
    if(userParameters.zero > 0.5) {
      _legController->_zeroEncoders = true;
    } else {
      _legController->_zeroEncoders = false;

      for(int leg(0); leg<4; ++leg){
        for(int jidx(0); jidx<3; ++jidx){

//          pTarget_12(leg*3 + jidx) = _legController->datas[leg].q[jidx] + 0.1 * (pTarget_12(leg*3 + jidx) - _legController->datas[leg].q[jidx]);
//          std::cout << "joint angle" << leg << jidx << ": " << _legController->datas[leg].q[jidx];
//          std::cout << "joint target" << leg << jidx << ": " << pTarget_12(leg*3 + jidx);
          _legController->commands[leg].qDes[jidx] = pTarget_12(leg*3 + jidx);
//          _legController->commands[leg].qDes[jidx] = qInitVec(leg*3 + jidx);
          _legController->commands[leg].qdDes[jidx] = 0.;
          _legController->commands[leg].tauFeedForward[jidx] = userParameters.tau_ff;
        }
        _legController->commands[leg].kpJoint = kpMat;
        _legController->commands[leg].kdJoint = kdMat;
      }
//      std::cout << std::endl;
    }
  }


}
