#ifndef INCLUDED_vec_op_hpp_
#define INCLUDED_vec_op_hpp_

#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

template <class T>
std::vector<T> operator+(const std::vector<T> &v1, const std::vector<T> &v2); // vector + vector
template <class T>
std::vector<T> operator-(const std::vector<T> &v1, const std::vector<T> &v2); // vector - vector
template <class T>
std::vector<T>& operator+=(std::vector<T> &v1, const std::vector<T> &v2); // vector += vector
template <class T>
std::vector<T>& operator-=(std::vector<T> &v1, const std::vector<T> &v2); // vector -= vector
template <class T>
std::vector<T>& operator*=(std::vector<T> &v, const T &c); // vector *= scalar
template <class T>
std::vector<T> operator*(const std::vector<T> &v, const T &c); // vector * scalar
template <class T>
std::vector<T> operator*(const T &c, const std::vector<T> &v); // scalar * vector
template <class T>
std::vector<std::vector<T>>& operator*=(std::vector<std::vector<T>> &m, const T &c); // matrix *= scalar
template <class T>
std::vector<std::vector<T>> operator*(const std::vector<std::vector<T>> &m, const T &c); // matrix * scalar
template <class T>
std::vector<std::vector<T>> operator*(const T &c, const std::vector<std::vector<T>> &m); // scalar * matrix

namespace vec_op {
  template <class T>
  T dot(const std::vector<T> &v1, const std::vector<T> &v2); // vector * vector: inner product
  template <class T>
  std::vector<T> dot(const std::vector<std::vector<T>> &m, const std::vector<T> &v); // matrix * vector
  template <class Tv, class Tm>
  Tm dot(const std::vector<Tv> &v, const std::vector<Tm> &m); // vector * matrix
  template <class Tv, class Tm>
  Tm paralleldot(const std::vector<Tv> &v, const std::vector<Tm> &m); // vector * matrix parallelized if OpenMP enabled
}




template <class T>
std::vector<T> operator+(const std::vector<T> &v1, const std::vector<T> &v2) {
  std::vector<T> ans = v1;
  for (size_t i = 0, size = ans.size(); i < size; ++i)
    ans[i] += v2[i];
  return ans;
}

template <class T>
std::vector<T> operator-(const std::vector<T> &v1, const std::vector<T> &v2) {
  std::vector<T> ans = v1;
  for (size_t i = 0, size = ans.size(); i < size; ++i)
    ans[i] -= v2[i];
  return ans;
}

template <class T>
std::vector<T>& operator+=(std::vector<T> &v1, const std::vector<T> &v2) {
  for (size_t i = 0, size = v1.size(); i < size; ++i)
    v1[i] += v2[i];
  return v1;
}

template <class T>
std::vector<T>& operator-=(std::vector<T> &v1, const std::vector<T> &v2) {
  for (size_t i = 0, size = v1.size(); i < size; ++i)
    v1[i] -= v2[i];
  return v1;
}

template <class T>
std::vector<T>& operator*=(std::vector<T> &v, const T &c) {
  for (T &e : v)
    e *= c;
  return v;
}

template <class T>
std::vector<T> operator*(const std::vector<T> &v, const T &c) {
  std::vector<T> ans = v;
  for (T &e : ans)
    e *= c;
  return ans;
}

template <class T>
std::vector<T> operator*(const T &c, const std::vector<T> &v) {
  std::vector<T> ans = v;
  for (T &e : ans)
    e *= c;
  return ans;
}

template <class T>
std::vector<std::vector<T>>& operator*=(std::vector<std::vector<T>> &m, const T &c) {
  for (std::vector<T> &me : m) {
    me *= c;
  }
  return m;
}

template <class T>
std::vector<std::vector<T>> operator*(const std::vector<std::vector<T>> &m, const T &c) {
  std::vector<std::vector<T>> ans = m;
  for (std::vector<T> & e : ans)
    e *= c;
  return ans;
}

template <class T>
std::vector<std::vector<T>> operator*(const T &c, const std::vector<std::vector<T>> &m) {
  std::vector<std::vector<T>> ans = m;
  for (std::vector<T> &e : ans)
    e *= c;
  return ans;
}


namespace vec_op {
  template <class T>
  T dot(const std::vector<T> &v1, const std::vector<T> &v2) {
    T ans(0);
    for (size_t i = 0, size = v1.size(); i < size; ++i)
      ans += v1[i] * v2[i];
    return ans;
  }

  template <class T>
  std::vector<T> dot(const std::vector<std::vector<T>> &m, const std::vector<T> &v) {
    std::vector<T> ans(m.size());
    for (size_t i = 0, size = m.size(); i < size; ++i) {
      ans[i] = dot(m[i], v);
    }
    return ans;
  }

  template <class Tv, class Tm>
  Tm dot(const std::vector<Tv> &v, const std::vector<Tm> &m) {
    Tm ans = m[0];
    for (size_t j = 0, jsize = m[0].size(); j < jsize; ++j) {
      ans[j] = v[0] * m[0][j];
      for (size_t i = 1, isize = m.size(); i < isize; ++i) {
	ans[j] += v[i] * m[i][j];
      }
    }
    return ans;
  }

  template <class Tv, class Tm>
  Tm paralleldot(const std::vector<Tv> &v, const std::vector<Tm> &m) {
    Tm ans = m[0];
    size_t jsize = m[0].size();
    size_t isize = m.size();
    
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t j = 0; j < jsize; ++j) {
      ans[j] = v[0] * m[0][j];
      for (size_t i = 1; i < isize; ++i) {
	ans[j] += v[i] * m[i][j];
      }
    }
    return ans;
  }
}

#endif
