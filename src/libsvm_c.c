/* 
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
*/

/* $Id$ */

#include <libsvm/svm.h>
#include <caml/mlvalues.h>
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/custom.h>
#include <caml/bigarray.h>
#include <caml/fail.h>

#include <stdio.h>

/* Taken from svm.cpp */
struct svm_model
{
  struct svm_parameter param;	/* parameter */
  int nr_class;		/* number of classes, = 2 in regression/one class svm */
  int l;			/* total #SV */
  struct svm_node **SV;		/* SVs (SV[l]) */
  double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[n-1][l]) */
  double *rho;		/* constants in decision functions (rho[n*(n-1)/2]) */
  double *probA;          /* pariwise probability information */
  double *probB;
  
  /* for classification only */
  
  int *label;		/* label of each class (label[n]) */
  int *nSV;		/* number of SVs for each class (nSV[n]) */
  /* nSV[0] + nSV[1] + ... + nSV[n-1] = l */
  /* XXX */
  int free_sv;		/* 1 if svm_model is created by svm_load_model */
  /* 0 if svm_model is created by svm_train */
};

struct svm_parameter param;
struct svm_problem prob;
struct svm_node *x_space;
double *target;
void finalize_svm_model(value m);
static struct custom_operations svm_model_ops =
  {
    "Libsvm.svm_model.v1.0",
    &finalize_svm_model,
    custom_compare_default,
    custom_hash_default,
    custom_serialize_default,
    custom_deserialize_default
  };

/* Set up one pattern. */
void setup_pattern(int l, int k, int idx, double* pattern){
  int i;
  x_space = (struct svm_node *) malloc((k+1)*sizeof(struct svm_node));
  for(i=0; i<k; ++i){
    x_space[i].index = i+1;
    x_space[i].value = pattern[idx+i*l];
  }
  x_space[k].index = -1;
  x_space[k].value = 42.0; /* ;) */
}

/* Destroy current pattern. */
void destroy_pattern(){
  free(x_space);
}

/* Set up svm parameter struct. */
void setup_param(int svm_type, int kernel_type, double degree, double gamma, double coef0,
		 int cache_size, double eps, double C, int nr_weight, int *weight_label, 
		 double* weight, double nu, double p, int shrinking, int probability){
  param.svm_type = svm_type;
  param.kernel_type = kernel_type;
  param.degree = degree;
  param.gamma = gamma;
  param.coef0 = coef0;
  param.cache_size = cache_size;
  param.eps = eps;
  param.C = C;
  param.nr_weight = nr_weight;
  param.weight_label = weight_label;
  param.weight = weight;
  param.nu = nu;
  param.p = p;
  param.shrinking = shrinking;
  param.probability = probability;

/*   printf("svm_type = %d\n", svm_type); */
/*   printf("kernel_type = %d\n", kernel_type); */
/*   printf("degree = %g\n", degree); */
/*   printf("gamma = %g\n", gamma); */
/*   printf("coef0 = %g\n", coef0); */
/*   printf("cache_size = %d\n", cache_size); */
/*   printf("eps = %g\n", eps); */
/*   printf("C = %g\n", C); */
/*   printf("nu = %g\n", nu); */
/*   printf("p = %g\n", p); */
/*   printf("shrinking = %d\n", shrinking); */
/*   printf("probability = %d\n", probability); */
}

/* Set up svm problem struct. */
void setup_problem(int l, int k, double *x, double *y){
 int i,j,pos,idx,num_zero;
 prob.l = l;
 prob.y = y;
 num_zero=0;
 for(i = 0; i < k*l; ++i){
   if(x[i] == 0.0)
     ++num_zero;
 }
   
 prob.x = (struct svm_node **)malloc(prob.l*sizeof(struct svm_node *));
 x_space = (struct svm_node *)malloc((((k+1)*l)-num_zero)*sizeof(struct svm_node));
 target = (double *)malloc(prob.l*sizeof(double));
 pos = 0; idx = 1;
 for(i = 0; i < l; ++i){
   prob.x[i] = &x_space[pos];
   for(j = 0; j < k; ++j){
     if(x[i+j*l] != 0.0){
       x_space[pos].index = idx;
       x_space[pos].value = x[i+j*l];
       ++pos;
     }
     ++idx;
   }
   x_space[pos].index = -1;
   x_space[pos].value = 42.0;
   ++pos;
   idx = 1;
 }

/*  for(i = 0; i < ((k+1)*l)-num_zero; ++i){ */
/*    printf("(%d,%g) ", x_space[i].index, x_space[i].value); */
/*    if(x_space[i].index == -1) */
/*      printf("\n"); */
/*  } */
}

/* Destroy svm problem struct. */
void destroy_problem(){
  free(prob.x);
  free(x_space);  
}

/* Setup confusion matrix */
void setup_cm(int nr_class, double *cm, double *target){
  int i;
  for(i=0; i<prob.l; ++i){
    ++cm[((int)target[i]-1)*nr_class+((int)prob.y[i]-1)];
  }
}

/* Copy an svm_model, for now by saving and loading. TODO: direct copy */
struct svm_model* copy_model(struct svm_model *model, int k){
  int i,j,pos,n,l,m,elements;
  struct svm_node *x_space;
  struct svm_model *copy;
  copy = (struct svm_model *) malloc(sizeof(struct svm_model));
  copy->param = model->param;
  copy->nr_class = model->nr_class;
  copy->l = model->l;
  l = model->l;
  n = model->nr_class;
  copy->SV = (struct svm_node **) malloc(l*sizeof(struct svm_node *));
  elements = l*(k+1);
  x_space = (struct svm_node *) malloc(elements*sizeof(struct svm_node));
  pos=0;
  for(i = 0; i<l; ++i){
    copy->SV[i] = &x_space[pos];
    for(j = 0; j<k; ++j){
      x_space[pos].index = model->SV[i][j].index;
      x_space[pos].value = model->SV[i][j].value;
      ++pos;
    }
    x_space[pos].index = -1;
    x_space[pos].value = 42.0;
    ++pos;
  }
  copy->sv_coef = (double **) malloc((n-1)*sizeof(double *));
  for(i = 0; i<n-1; ++i){
    copy->sv_coef[i] = (double *) malloc(l*sizeof(double));
    for(j = 0; j<l; ++j){
      copy->sv_coef[i][j] = model->sv_coef[i][j];
    }
  }
  m = (n*(n-1))/2;
  copy->rho = (double *) malloc(m*sizeof(double));
  for(i = 0; i<m; ++i){
    copy->rho[i] = model->rho[i];
  }
  copy->probA = NULL;
  if(model->probA != NULL){
    copy->probA = (double *) malloc(m*sizeof(double));
    for(i = 0; i<m; ++i){
      copy->probA[i] = model->probA[i];
    }
  }
  copy->probB = NULL;
  if(model->probB != NULL){
    copy->probB = (double *) malloc(m*sizeof(double));
    for(i = 0; i<m; ++i){
      copy->probB[i] = model->probB[i];
    }
  }
  copy->label = (int *) malloc(n*sizeof(int));
  for(i = 0; i<n; ++i){
    copy->label[i] = model->label[i];
  }
  copy->nSV = (int *) malloc(n*sizeof(int));
  for(i = 0; i<n; ++i)
    copy->nSV[i] = model->nSV[i];
  copy->free_sv = 1;

  /* Check */
  svm_save_model("xxxtmp.model",model);   
  svm_save_model("xxxtmp2.model",copy);   
/*   svm_save_model("xxxtmp.model",model);    */
/*   svm_destroy_model(model);  */
/*   return svm_load_model("xxxtmp.model");  */
  return copy;
}

void finalize_svm_model(value m){
  svm_destroy_model(*((struct svm_model **) Data_custom_val (m)));
}

CAMLprim value svm_train_c(value x, value y, value p){
  CAMLparam3 (x,y,p);
  CAMLlocal1 (m);
  /* TODO: Outfactor to common setup function. */
  const char *msg;
  int *weight_label;
  double *weight;
  int k;
  k = Bigarray_val(x)->dim[1];
  if(Long_val(Field(p,8)) == 0){
    weight_label = NULL;
    weight = NULL;
  }
  else{
    weight_label = (int *) Data_bigarray_val(Field(p,9)); 
    weight = (double *) Data_bigarray_val(Field(p,10));
  }
  setup_param(Long_val(Field(p,0)),
	      Long_val(Field(p,1)),
	      Double_val(Field(p,2)),
	      Double_val(Field(p,3)),
	      Double_val(Field(p,4)),
	      Long_val(Field(p,5)),
	      Double_val(Field(p,6)),
	      Double_val(Field(p,7)),
	      Long_val(Field(p,8)),
	      weight_label,
	      weight,
	      Double_val(Field(p,11)),
	      Double_val(Field(p,12)),
	      Bool_val(Field(p,13)),
	      Bool_val(Field(p,14)));
  setup_problem(Bigarray_val(x)->dim[0], 
		Bigarray_val(x)->dim[1], 
		(double *) Data_bigarray_val(x),
		(double *) Data_bigarray_val(y));
  msg=svm_check_parameter(&prob,&param);
  if(msg != NULL)
    failwith((char *) msg);
  m = alloc_custom(&svm_model_ops, sizeof(struct svm_model *), 1, 10);
  *((struct svm_model **) Data_custom_val(m)) = copy_model(svm_train(&prob,&param),k);
  destroy_problem();
  CAMLreturn (m);
}

CAMLprim void svm_predict_c(value m, value x, value plabels){
  CAMLparam3 (m,x,plabels);
  int idx,l,k;
  double* p;
  l=Bigarray_val(x)->dim[0];
  k=Bigarray_val(x)->dim[1];
  p=(double *) Data_bigarray_val(plabels);
  for(idx=0; idx<l; ++idx){
    setup_pattern(l, k, idx,
		  (double *) Data_bigarray_val(x));
    p[idx] = svm_predict(*((struct svm_model **) Data_custom_val(m)), x_space);
    destroy_pattern();
  }
  CAMLreturn0;
}

CAMLprim void svm_cross_validation_c(value x, value y, value p, value nr_fold, value cm){
  CAMLparam5 (x,y,p,nr_fold,cm);
  const char *msg=NULL;
  int *weight_label;
  double *weight;
  if(Long_val(Field(p,8)) == 0){
    weight_label = NULL;
    weight = NULL;
  }
  else{
    weight_label = (int *) Data_bigarray_val(Field(p,9)); 
    weight = (double *) Data_bigarray_val(Field(p,10));
  }
  setup_param(Long_val(Field(p,0)),
	      Long_val(Field(p,1)),
	      Double_val(Field(p,2)),
	      Double_val(Field(p,3)),
	      Double_val(Field(p,4)),
	      Long_val(Field(p,5)),
	      Double_val(Field(p,6)),
	      Double_val(Field(p,7)),
	      Long_val(Field(p,8)),
	      weight_label,
	      weight,
	      Double_val(Field(p,11)),
	      Double_val(Field(p,12)),
	      Bool_val(Field(p,13)),
	      Bool_val(Field(p,14)));
  setup_problem(Bigarray_val(x)->dim[0], 
		Bigarray_val(x)->dim[1], 
		(double *) Data_bigarray_val(x),
		(double *) Data_bigarray_val(y));
  msg=svm_check_parameter(&prob,&param);
  if(msg != NULL)
    failwith((char *) msg);
  svm_cross_validation(&prob,&param,Long_val(nr_fold),target);
  setup_cm(Bigarray_val(cm)->dim[0],(double *) Data_bigarray_val(cm), target);
  destroy_problem();
  free(target);
  CAMLreturn0;
}

CAMLprim value svm_get_svm_type_c(value m){
  CAMLparam1 (m);
  CAMLreturn (Val_long(svm_get_svm_type(*((struct svm_model **) Data_custom_val(m)))));
}

CAMLprim value svm_get_nr_class_c(value m){
  CAMLparam1 (m);
  CAMLlocal1 (r);
  r = Val_long(svm_get_nr_class(*((struct svm_model **) Data_custom_val(m))));
  CAMLreturn (r);
}

CAMLprim void svm_get_labels_c(value m, value l){
  CAMLparam2 (m,l);
  svm_get_labels(*((struct svm_model **) Data_custom_val(m)), 
		 (int *) Data_bigarray_val(l));
  CAMLreturn0;
}

CAMLprim value svm_get_svr_probability_c(value m){
  CAMLparam1 (m);
  CAMLreturn (copy_double(svm_get_svr_probability(*((struct svm_model **) Data_custom_val(m)))));
}

CAMLprim void svm_predict_values_c(value m, value x, value dv){
  CAMLparam3 (m,x,dv);
  setup_pattern(Bigarray_val(x)->dim[0],
		Bigarray_val(x)->dim[1],
		0,
		(double *) Data_bigarray_val(x));
  svm_predict_values(*((struct svm_model **) Data_custom_val(m)),
		     x_space,
		     (double *) Data_bigarray_val(dv));
  destroy_pattern();
  CAMLreturn0;
}

CAMLprim void svm_predict_probability_c(value m, value x, value pe, value plabels){
  CAMLparam4 (m,x,pe,plabels);
  int idx,l,k,nr_class;
  double *p; double *ppe;
  l=Bigarray_val(x)->dim[0];
  k=Bigarray_val(x)->dim[1];
  p = (double *) Data_bigarray_val(plabels);
  ppe = (double *) Data_bigarray_val(pe);
  nr_class = svm_get_nr_class(*((struct svm_model **) Data_custom_val(m)));
  for(idx=0; idx<l; ++idx){
    setup_pattern(l, k, idx, 
		  (double *) Data_bigarray_val(x));
    p[idx] = svm_predict_probability(*((struct svm_model **) Data_custom_val(m)), 
				     x_space, 
				     &ppe[idx*nr_class]);
    destroy_pattern();
  }
  CAMLreturn0;
}

CAMLprim value svm_check_probability_model_c(value m){
  CAMLparam1 (m);
  CAMLlocal1 (r);
  if(svm_check_probability_model(*((struct svm_model **) Data_custom_val(m))))
    r = Val_true;
  else
    r = Val_false;
  CAMLreturn (r);
}

CAMLprim void svm_save_model_c(value file, value m){
  CAMLparam2 (file,m);
  if(svm_save_model(String_val(file), *((struct svm_model **) Data_custom_val(m))))
    failwith("Could not save model");
  CAMLreturn0;
}

CAMLprim value svm_load_model_c(value file){
  CAMLparam1 (file);
  CAMLlocal1 (m);
  m = alloc_custom(&svm_model_ops, sizeof(struct svm_model *), 1, 10);
  *((struct svm_model **) Data_custom_val(m)) = svm_load_model(String_val(file));
  if(*((struct svm_model **) Data_custom_val(m)) == NULL)
    failwith("Could not load model");
  CAMLreturn (m);
}

