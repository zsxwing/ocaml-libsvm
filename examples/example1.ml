(* D. Brugger, September 2005
   example1.ml - simple example for usage of ocaml-libsvm bindings.
   $Id$
*)

open Lacaml.D
open Libsvm

(** map_cols applies function f to all columns of matrix a. *)
  let map_cols f a =
    let n = Mat.dim2 a in 
    let r = Mat.create 1 n in
      for i = 1 to n do
	r.{1,i} <- f (Mat.col a i)
      done;
      r
  ;;
(** sum computes the sum of the elements in each column of matrix a. *)
let sum = map_cols Vec.sum
  (* let sum = fold (+.) 0. *)
;;
(** mean computes the mean of the elements in each column of matrix a. *)
let mean a = 
  let m = float_of_int (Mat.dim1 a) in
    Mat.map (fun x -> x /. m) (sum a)
;;

(* Some random training data. 100 patterns with 3 attributes. *)
let x = Mat.random 100 3
;;
(* Some test data. *)
let xt = Mat.random 50 3
;;

(* Induce correlation between patterns and class labels. *)
let y = Mat.map (fun x -> if x < 0.5 then 1. else -.1.)
  (Mat.transpose (mean (Mat.transpose x)))
;;

let _ =
  (* Train svm model with standard parameters. *)
  let m = Libsvm.svm_train x y in
    (* Predict class labels. *)
  let py = Libsvm.svm_predict m xt in
    (* Compute expected class labels for testing. *)
  let ey = Mat.map (fun x -> if x < 0.5 then 1. else -.1.)
    (Mat.transpose (mean (Mat.transpose xt))) in
  let n = Mat.dim1 py in
  let total_correct = ref 0 in
    (* Compare expected and predicted class labels. *)
    for i = 1 to n do
      if py.{i,1} = ey.{i,1}
      then incr total_correct;
    done;
    Printf.printf "Total correct = %d\n" !total_correct; 
    Printf.printf "Fraction correct = %g%%\n" ((float_of_int !total_correct) /.
      (float_of_int n) *. 100.0)
;;
