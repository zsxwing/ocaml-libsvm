(*
 $Id$
 D. Brugger, 22 September 2005
 libsvm/libsvm.mli - Interface for libsvm bindings.
*)

module Libsvm :
  sig
    type mat =
      (float, Bigarray.float64_elt, Bigarray.fortran_layout)
      Bigarray.Array2.t
    type mat_int =
      (int32, Bigarray.int32_elt, Bigarray.fortran_layout) Bigarray.Array2.t
    type vec =
      (float, Bigarray.float64_elt, Bigarray.fortran_layout)
      Bigarray.Array1.t
    type vec_int =
      (int32, Bigarray.int32_elt, Bigarray.fortran_layout) Bigarray.Array1.t
    type svm_type_t = C_SVC | NU_SVC | ONE_CLASS | EPSILON_SVR | NU_SVR
    type kernel_type_t = LINEAR | POLY | RBF | SIGMOID
    type svm_parameter_t = {
      svm_type : svm_type_t;
      kernel_type : kernel_type_t;
      degree : float;
      gamma : float;
      coef0 : float;
      cache_size : int;
      eps : float;
      c : float;
      nr_weight : int;
      weight_label : mat_int;
      weight : mat;
      nu : float;
      p : float;
      shrinking : bool;
      probability : bool;
    }
	(** Abstract datatype which represents a svm_model. *)
    type svm_model_t
    external svm_get_svm_type : m:svm_model_t -> svm_type_t
      = "svm_get_svm_type_c"
	(** This function returns the svm type which was used to train the
	    model. 
	    @param m The svm model.
	    @return Svm type. 
	*)
    external svm_get_nr_class : m:svm_model_t -> int = "svm_get_nr_class_c"
	(** "For a classification model, this function gives the
	    number of classes. For a regression or an one-class model, 2
	    is returned." (from libsvm-2.8 README) 
	    @param m The svm model.
	    @return The numer of classes.
	*)
    external svm_check_probability_model : m:svm_model_t -> bool
      = "svm_check_probability_model_c"
	(** "This function checks whether the model contains required
	    information to do probability estimates. If so, it returns
	    true. Otherwise, false is returned. This function should be called
	    before calling svm_get_svr_probability and
	    svm_predict_probability." (from libsvm-2.8 README abridged)
	    @param m The svm model.
	    @return See description.
	*)
    external svm_save_model : file:string -> m:svm_model_t -> unit
      = "svm_save_model_c"
	(** "This function saves a model to a file." (from libsvm-2.8 README abridged).
	    Raises execption Failure if not successful.
	    @param m The svm model.
	    @param file The filename.
	*)
    external svm_load_model : file:string -> svm_model_t = "svm_load_model_c"
	(** Loads a svm model stored in a file.
	    @param file The filename.
	    @return The svm model.
	*)
    val svm_train :
      ?svm_type:svm_type_t ->
      ?kernel_type:kernel_type_t ->
      ?degree:float ->
      ?gamma:float ->
      ?coef0:float ->
      ?cache_size:int ->
      ?eps:float ->
      ?c:float ->
      ?weights:(int32 * float) list ->
      ?nu:float ->
      ?p:float ->
      ?shrinking:bool ->
      ?probability:bool -> Lacaml_float64.mat -> mat -> svm_model_t
	(** "This function constructs and returns an SVM model according to
	    the given training data and parameters." (form libsvm-2.8 README abridged)

	    @param svm_type The svm type. Possible values are:
	    C_SVC:		C-SVM classification
	    NU_SVC:		nu-SVM classification
	    ONE_CLASS:		one-class-SVM
	    EPSILON_SVR:	epsilon-SVM regression
	    NU_SVR:		nu-SVM regression
	    @param kernel_type The kernel function type. Possible values are:
	    LINEAR:	u'*v
	    POLY:	(gamma*u'*v + coef0)^degree
	    RBF:	exp(-gamma*|u-v|^2)
	    SIGMOID:	tanh(gamma*u'*v + coef0)
	    @param degree The degree for polynomial kernel.
	    @param gamma The parameter gamma for polynomial,rbf and sigmoid kernels.
	    @param coef0 The parameter coef0 for polynomial and sigmoid kernels.
	    @param cache_size The cache size of the kernel cache in MB.
	    @param eps Stopping criteria.
	    @param C The parameter C for C_SVC, EPSILON_SVR and NU_SVR.
	    @param nu The parameter nu gor NU_SVC, ONE_CLASS and NU_SVR.
	    @param p The parameter p for EPSILON_SVR
	    @param weight Association list of (label,weight) for changing penalty for 
	    classes in C_SVC.
	    @param shrinking Whether to use the shrinking heuristics.
	    @param probability Whether to perform probability estimates.
	    @param x The training data.
	    @param y The target vector.
	    @return The svm model.
	*)
    val svm_predict : m:svm_model_t -> x:mat -> mat
      (** "This function does classification or regression on a test vector x
	  given a model.

	  For a classification model, the predicted class for x is returned.
	  For a regression model, the function value of x calculated using
	  the model is returned. For an one-class model, +1 or -1 is
	  returned." (from libsvm-2.8 README) 
	  @param m The svm model.
	  @param x The test vector.
	  @return Label or function value.
      *)
    val svm_cross_validation :
      ?svm_type:svm_type_t ->
      ?kernel_type:kernel_type_t ->
      ?degree:float ->
      ?gamma:float ->
      ?coef0:float ->
      ?cache_size:int ->
      ?eps:float ->
      ?c:float ->
      ?weights:'a list ->
      ?nu:float ->
      ?p:float ->
      ?shrinking:bool ->
      ?probability:bool ->
      Lacaml_float64.mat -> Lacaml_float64.mat -> int -> Lacaml_float64.mat
      (** "This function conducts cross validation. Data are separated to
	  nr_fold folds. Under given parameters, sequentially each fold is
	  validated using the model from training the remaining. Predicted
	  labels in the validation process are stored in the array called
	  target." (from libsvm-2.8 README)
	  
	  For a description of parameters. See description of svm_train.

	  @param x The training data.
	  @param y The target vector.
	  @param nr_fold The fold.
	  @return The confusion matrix.
      *)
    val svm_get_labels :
      svm_model_t ->
      (int32, Bigarray.int32_elt, Bigarray.fortran_layout) Bigarray.Array2.t
      (** "For a classification model, this function outputs the name of
	  labels into an array called label. For regression and one-class
	  models, label is unchanged." (from libsvm-2.8 README)
	  @param m The svm model.
	  @return Bigarray with labels.
      *)
    val svm_get_svr_probability : svm_model_t -> float
	(** "For a regression model with probability information, this function
	    outputs a value sigma > 0. For test data, we consider the
	    probability model: target value = predicted value + z, z: Laplace
	    distribution e^(-|z|/sigma)/(2sigma)"  (from libsvm-2.8 README)

	    If the model is not for svr or does not contain the required information
	    the exception Failure is raised.

	    @param m The svm model.
	    @return The value sigma.
	*)
    val svm_predict_values :
      svm_model_t ->
      mat ->
      (float, Bigarray.float64_elt, Bigarray.fortran_layout)
      Bigarray.Array2.t
      (** " This function gives decision values on a test vector x given a
	  model.

	  For a classification model with nr_class classes, this function
	  gives nr_class*(nr_class-1)/2 decision values in the array
	  dec_values, where nr_class can be obtained from the function
	  svm_get_nr_class. The order is label[0] vs. label[1], ...,
	  label[0] vs. label[nr_class-1], label[1] vs. label[2], ...,
	  label[nr_class-2] vs. label[nr_class-1], where label can be
	  obtained from the function svm_get_labels."  (from libsvm-2.8 README)
	  
	  @param m The svm model.
	  @param x The data.
	  @return The decision values.
      *)
    val svm_predict_probability :
      svm_model_t ->
      mat ->
      (float, Bigarray.float64_elt, Bigarray.fortran_layout)
	Bigarray.Array2.t *
	(float, Bigarray.float64_elt, Bigarray.fortran_layout)
	Bigarray.Array2.t
	(** "This function does classification or regression on a test vector x
	    given a model with probability information.
	    
	    For a classification model with probability information, this
	    function gives nr_class probability estimates in the array
	    prob_estimates. nr_class can be obtained from the function
	    svm_get_nr_class. The class with the highest probability is
	    returned. For all other situations, the array prob_estimates is
	    unchanged and the returned value is the same as that of
	    svm_predict."  (from libsvm-2.8 README)
	    
	    @param m The svm model.
	    @param x The test data.
	    @param The probability estimates.
	*)
    val svm_scale :
      ?lower:float ->
      ?upper:float -> Lacaml_float64.mat -> Lacaml_float64.mat
	(** This function should be applied to the training and test data
	    *before* using svm_train/svm_crossvalidatio etc., because of
	    numerical issues.

	    It scales all features to the range [+1,-1].
	    
	    @param lower The lower bound.
	    @param upper The upper bound.
	    @param x The training/test data.
	*)
      val accuracy : Lacaml_float64.mat -> float
      val tcr : Lacaml_float64.mat -> Lacaml_float64.vec
      val gm2 : Lacaml_float64.mat -> float
  end
