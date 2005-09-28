(* 
   ocaml-libsvm - OCaml bindings to libsvm
   Copyright (C) 2005 Dominik Brugger

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*)

(* $Id$ *)

open Bigarray
open Lacaml.D

module Libsvm =
struct
  (* Some useful definitions from ../util/matlab.ml *)
  module Matlab =
  struct
    let map2 f a1 a2 =
      let m1 = Mat.dim1 a1 
      and n1 = Mat.dim2 a1
      and m2 = Mat.dim1 a2
      and n2 = Mat.dim2 a2 in
	if m1 == m2 || n1 == n2 then
	  begin
	    let r = Mat.create m1 n1 in
	      for i = 1 to (max m1 m2) do
		for j = 1 to (max n1 n2) do
		  r.{i,j} <- f a1.{(min i m1),(min j n1)} a2.{(min i m2),(min j n2)};
		done;
	      done;
	      r
	  end
	else
	  failwith "Incompatible matrix dimensions"
    let sub = map2 ( -. )
    ;;
    let add = map2 ( +. )
    ;;
    let mul = map2 ( *. )
    ;;
    let div = map2 ( /. )
    ;;
    let map_cols f a =
      let n = Mat.dim2 a in 
      let r = Mat.create 1 n in
	for i = 1 to n do
	  r.{1,i} <- f (Mat.col a i)
	done;
	r
    ;;
    let max = map_cols (Vec.max)
    ;;
    let min = map_cols (Vec.min)
    ;;
    let sum = map_cols Vec.sum
    ;;
  end
  type mat = (float, Bigarray.float64_elt, Bigarray.fortran_layout) Bigarray.Array2.t
  type mat_int = (int32, Bigarray.int32_elt, Bigarray.fortran_layout) Bigarray.Array2.t
  type vec = (float, Bigarray.float64_elt, Bigarray.fortran_layout) Bigarray.Array1.t
  type vec_int = (int32, Bigarray.int32_elt, Bigarray.fortran_layout) Bigarray.Array1.t
  type svm_type_t = C_SVC | NU_SVC | ONE_CLASS | EPSILON_SVR | NU_SVR
  type kernel_type_t = LINEAR | POLY | RBF | SIGMOID
  type svm_parameter_t = { svm_type:svm_type_t;
			   kernel_type:kernel_type_t;
			   degree:float;
			   gamma:float;
			   coef0:float;
			   cache_size:int;
			   eps:float;
			   c:float;
			   nr_weight:int;
			   weight_label:mat_int;
			   weight:mat;
			   nu:float;
			   p:float;
			   shrinking:bool;
			   probability:bool }
  type svm_model_t
  external svm_train_c :
    x:mat -> y:mat -> p:svm_parameter_t -> svm_model_t
    = "svm_train_c"
  ;;
  external svm_predict_c :
    m:svm_model_t -> x:mat -> plabels:mat -> unit
    = "svm_predict_c"
  ;;
  external svm_cross_validation_c : 
    x:mat -> y:mat -> p:svm_parameter_t -> nr_fold:int -> cm:mat -> unit
    = "svm_cross_validation_c"
  ;;
  external svm_get_svm_type :
    m:svm_model_t -> svm_type_t
    = "svm_get_svm_type_c"
  ;;
  external svm_get_nr_class :
    m:svm_model_t -> int
    = "svm_get_nr_class_c"
  ;;
  external svm_get_labels_c :
    m:svm_model_t -> l:mat_int -> unit 
    = "svm_get_labels_c"
  ;;
  external svm_get_svr_probability_c :
    m:svm_model_t -> float
    = "svm_get_svr_probability_c"
  ;;
  external svm_predict_values_c :
    m:svm_model_t -> x:mat -> dv:mat -> unit 
    = "svm_predict_values_c"
  ;;
  external svm_predict_probability_c :
    m:svm_model_t -> x:mat -> pe:mat -> plabels:mat -> unit
    = "svm_predict_probability_c"
  ;;
  external svm_check_probability_model :
    m:svm_model_t -> bool
    = "svm_check_probability_model_c"
  ;;
  external svm_save_model :
    file:string -> m:svm_model_t -> unit
    = "svm_save_model_c"
  ;;
  external svm_load_model :
    file:string -> svm_model_t
    = "svm_load_model_c"
  ;;
  let svm_train
    ?(svm_type = C_SVC) 
    ?(kernel_type = RBF)
    ?(degree = 3.0)
    ?(gamma = nan)
    ?(coef0 = 0.0)
    ?(cache_size = 40)
    ?(eps = 0.001)
    ?(c = 1.0)
    ?(weights = []) (* (int * float) list *)
    ?(nu = 0.5)
    ?(p = 0.1) 
    ?(shrinking = true)
    ?(probability = false) x y =
    let k = Lacaml.D.Mat.dim2 x in
    let gamma = 
      if (classify_float nan) = FP_nan
      then 1. /. (float_of_int k)
      else gamma in
    let nr_weight = List.length weights in
    let (il,wl) = List.split weights in
    let weight_label = (Bigarray.Array2.create int32 Bigarray.fortran_layout nr_weight 1) in
    let weight = (Bigarray.Array2.create float64 Bigarray.fortran_layout nr_weight 1) in
      for i = 1 to nr_weight do
	weight_label.{1,i} <- List.nth il (i-1);
	weight.{1,i} <- List.nth wl (i-1);
      done;
    let p = { svm_type = svm_type;
	      kernel_type = kernel_type;
	      degree = degree;
	      gamma = gamma;
	      coef0 = coef0;
	      cache_size = cache_size;
	      eps = eps;
	      c = c;
	      nr_weight = nr_weight;
	      weight_label = weight_label;
	      weight = weight;
	      nu = nu;
	      p = p;
	      shrinking = shrinking;
	      probability = probability }
    in
      svm_train_c ~x:x ~y:y ~p:p
  ;;
  let svm_predict ~m ~x =
    let l = Lacaml.D.Mat.dim1 x in
    let plabels = (Bigarray.Array2.create float64 Bigarray.fortran_layout l 1) in
      svm_predict_c m x plabels;
      plabels
  ;;
  let svm_cross_validation
    ?(svm_type = C_SVC) 
    ?(kernel_type = RBF)
    ?(degree = 3.0)
    ?(gamma = nan)
    ?(coef0 = 0.0)
    ?(cache_size = 40)
    ?(eps = 0.001)
    ?(c = 1.0)
    ?(weights = []) (* (int * float) list *)
    ?(nu = 0.5)
    ?(p = 0.1) 
    ?(shrinking = true)
    ?(probability = false) x y nr_fold =
    let k = Lacaml.D.Mat.dim2 x in
    let gamma = 
      if (classify_float nan) = FP_nan
      then 1. /. (float_of_int k)
      else gamma in
    let p = { svm_type = svm_type;
	      kernel_type = kernel_type;
	      degree = degree;
	      gamma = gamma;
	      coef0 = coef0;
	      cache_size = cache_size;
	      eps = eps;
	      c = c;
	      nr_weight = 0;
	      weight_label = (Bigarray.Array2.create int32 Bigarray.fortran_layout 0 0);
	      weight = (Bigarray.Array2.create float64 Bigarray.fortran_layout 0 0);
	      nu = nu;
	      p = p;
	      shrinking = shrinking;
	      probability = probability } 
    in
    let tmp = Matlab.max (Lacaml.D.Mat.transpose (Matlab.max y)) in
    let nr_class = int_of_float tmp.{1,1} in
    let cm = Lacaml.D.Mat.make nr_class nr_class 0. in
      svm_cross_validation_c ~x:x ~y:y ~p:p ~nr_fold:nr_fold  ~cm:cm;
      cm
  ;;
  let svm_get_labels m =
    let nr_class = svm_get_nr_class m in
    let l = (Bigarray.Array2.create int32 Bigarray.fortran_layout nr_class 1) in
      svm_get_labels_c m l;
      l
  ;;
  let svm_get_svr_probability m =
    match (svm_get_svm_type m) with
	EPSILON_SVR -> svm_get_svr_probability_c m
      | NU_SVR -> svm_get_svr_probability_c m
      | _ -> failwith "Model is not a regression model"
  ;;
  let svm_predict_values m x =
    let nr_class = svm_get_nr_class m in
    let num_dv = nr_class*(nr_class-1)/2 in
    let dv = (Bigarray.Array2.create float64 Bigarray.fortran_layout num_dv 1) in
      svm_predict_values_c m x dv;
      dv
  ;;
  let svm_predict_probability m x =
    if not (svm_check_probability_model m)
    then failwith "Model does not have probability information"
    else
      let l = Lacaml.D.Mat.dim1 x in
      let nr_class = svm_get_nr_class m in
      let pe = (Bigarray.Array2.create float64 Bigarray.fortran_layout nr_class l) in
      let plabels = (Bigarray.Array2.create float64 Bigarray.fortran_layout l 1) in
      svm_predict_probability_c m x pe plabels;
	(plabels,pe)
  ;;
  let svm_scale ?(lower=(-.1.0)) ?(upper=(1.0)) x =
    let feat_min = Matlab.min x 
    and feat_max = Matlab.max x in
      Lacaml.D.Mat.map (fun v -> v +. lower) (Lacaml.D.Mat.map (fun v -> v *. (upper -. lower)) (Matlab.div (Matlab.sub x feat_min) (Matlab.sub feat_max feat_min)))
  ;;
  let accuracy cm =
    let tmp = Matlab.sum (Mat.transpose (Matlab.sum cm)) in
      (Vec.sum (Mat.diag cm)) /. tmp.{1,1}
  ;;
  let tcr cm =
    Mat.diag (Matlab.div cm (Mat.transpose (Matlab.sum (Mat.transpose cm))))
  ;;
  let gm2 cm =
    let t = tcr cm in
    let n = Vec.dim t in
      Vec.prod t ** (1. /. (float_of_int n))
  ;;
end
;;

(* module C = *)
(* struct *)
(*   type atype = A | B | C *)
(*   let ( - ) x y = *)
(*     match (x,y) with *)
(* 	(A,A) -> B *)
(*       | (B,C) -> A *)
(*       | _ -> C     *)
(* end *)

(* module A = *)
(* struct  *)
(*   type atype = { a:int; b:int} *)
(*   let tmp () = { a=2; b=3} *)
(* end *)

(* module B = *)
(* struct *)
(*   type btype = { a:int; c:int} *)
(*   let tmp () = { a=4; c=2} *)
(* end *)
