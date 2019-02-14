/**
 * liangliang.pan
 */

#pragma once

/**
 *  Introduction to Kalman Filter
 * 
 * Prediction step:
 * 
 * equ(1) 	x' = F*x + u 		=> F:  State transistion matrix; x: state vector; u: noise 
 * equ(2) 	P' = F*P*F.t() + Q	=> P:	Covariance matrix of system states
 * 					=> Q:	Process noise, general is Identity matrix
 * Update step:
 * 
 * equ(3)	y = z - H*x'	=> The diff between observed measures z and the predicted value x'
 * 				=> H: Measurement Matrix
 * equ(4)	S = H*P'*H.t() + R
 * equ(5)	K = P' * H.t() * S.inverse() 
 * equ(6)	x = x' + K*y	=> K:Kalman Gain
 * equ(7)	P = (I - KH)*P' => 
 * 
 */


#include "Eigen/Dense"
#include <iostream>

namespace msckf_vio {

  class KalmanFilter{
  
  public:

  KalmaFilter(){is_initialized_ = false;}
  
  ~KalmanFilter();
  
  Eigen::VectorXd getX(){return x_;}
  bool isInitialized(){ return is_initialized_; }
  
  void initialization(Eigen::VectorXd x_in){ 
    x_ = x_in;
    is_initialized_ = true;
  }
  
  void setF(Eigen::MatrixXd F_in){ F_ = F_in;}
  void setP(Eigen::MatrixXd P_in){ P_ = P_in;}
  void setQ(Eigen::MatrixXd Q_in){ Q_ = Q_in;}
  void setH(Eigen::MatrixXd H_in){ H_ = H_in;}
  void setR(Eigen::MatrixXd R_in){ R_ = R_in;}
  
  void prediction(){
    x_ = F_ * x_;
    P_ = F_*P_*F_.transpose() + Q_;
  }
  
  void measurementUpdate(const Eigen::VectorXd &z){
    Eigen::VectorXd y = z - H_*x_;
    Eigen::MatrixXd S = H_*P_*H_.transpose() + R_;
    Eigen::MatrixXd K = P_*H_.transpose() * S.inverse();
    x_ = x_ + (K*y);
    Eigen::MatrixXd Ide_mat = Eigen::MatrixXd::Identity(x_.size(), x_.size());
    P_ = (Ide_mat - K*H_) * P_;
  }
  
  
  private:
    
    bool is_initialized_;
    
    // state vector
    Eigen::VectorXd x_;
  
    // State transistion matrix
    Eigen::MatrixXd F_;
    
    // state covariance matrix
    Eigen::MatrixXd P_;
    
    // process covariance matrix
    Eigen::MatrixXd Q_;
      
    // measurement matrix
    Eigen::MatrixXd H_;
    
    // measurement covariance matrix
    Eigen::MatrixXd R_;
    
  };
  

} // namespace msckf_vio

// example

int main(){
  double m_x = 0.0, m_y = 0.0;
  double last_timestamp = 0.0, now_timestamp = 0.0;
  msckf_vio::KalmanFilter kf;
  
  while(getData(m_x, m_y, now_timestamp)){
    // initial kalman filter
    if(!kf.isInitialized()){
      last_timestamp = now_timestamp;
      Eigen::VectorXd x_in(4, 1);
      x_in << m_x, m_y, 0.0, 0.0;
      kf.initialization(x_in);
      
      // state covariance matrix
      Eigen::MatrixXd P_in(4, 4);
      P_in << 1.0, 0.0, 0.0, 0.0,
	      0.0, 1.0, 0.0, 0.0,
	      0.0, 0.0, 100.0, 0.0,
	      0.0, 0.0, 0.0, 100.0;
      kf.setP(P_in);
      
      Eigen::MatrixXd Q_in(4, 4);
      Q_in << 1.0, 0.0, 0.0, 0.0,
	      0.0, 1.0, 0.0, 0.0,
	      0.0, 0.0, 1.0, 0.0,
	      0.0, 0.0, 0.0, 1.0;
      kf.setQ(Q_in);
      
      Eigen::MatrixXd H_in(2, 4);
      H_in << 1.0, 0.0, 0.0, 0.0,
	      0.0, 1.0, 0.0, 0.0;
      kf.setH(H_in);
      
      // Measurement covariance matrix
      // R need provided by Sensor supplier
      Eigen::MatrixXd R_in(2, 2);
      R_in << 0.0225, 0.0, 0.0, 0.0225;
      kf.setR(R_in);
    }
    double delta_t = now_timestamp - last_timestamp;
    last_timestamp = now_timestamp;
    Eigen::MatrixXd F_in(4, 4);
    F_in << 1.0, 0.0, delta_t, 0.0,
	    0.0, 1.0, 0.0, delta_t,
	    0.0, 0.0, 1.0, 0.0,
	    0.0, 0.0, 0.0, 1.0;
    kf.setF(F_in);
    
    kf.prediction();
    Eigen::VectorXd z(2, 1);
    z << m_x, m_y;
    kf.measurementUpdate(z);
    
    // get result
    Eigen::VectorXd x_out = kf.getX();
    std::cout << "Result of kalman filter = " << x_out(0) << ", " << x_out(1) << std::endl;
  }
}
