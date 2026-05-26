#include "decomposition/SVD/PCA.hpp"
#include "decomposition/SVD/SVDInput.hpp"
#include "mbs/attribution/AttributionRequest.hpp"
#include "mbs/attribution/AttributionResult.hpp"
#include "mbs/attribution/CrossSecondOrderCalculator.hpp"
#include "mbs/attribution/MatrixHessianCalculator.hpp"
#include "mbs/attribution/ScalarSecondOrderCalculator.hpp"
#include "mbs/factors/FactorBasis.hpp"
#include "mbs/factors/FactorKey.hpp"
#include "mbs/factors/FactorMove.hpp"
#include "mbs/numerics/FiniteDifference.hpp"
#include "mbs/numerics/Interpolation.hpp"
#include "mbs/rates/Bootstrap.hpp"
#include "mbs/rates/CurveCalibration.hpp"
#include "mbs/rates/DiscountCurve.hpp"
#include "mbs/rates/LmmSofr.hpp"
#include "mbs/rates/SwaptionVolProxy.hpp"
#include "mbs/scenarios/ScenarioLattice.hpp"
#include "mbs/scenarios/ScenarioPrice.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

decomposition::SVD::Matrix make_matrix(
    const std::vector<std::vector<double>> &rows) {
  if (rows.empty() || rows.front().empty()) {
    throw std::invalid_argument("matrix must have at least one row and column");
  }
  const std::size_t row_count = rows.size();
  const std::size_t col_count = rows.front().size();
  std::vector<double> values;
  values.reserve(row_count * col_count);
  for (const auto &row : rows) {
    if (row.size() != col_count) {
      throw std::invalid_argument("matrix rows must have consistent widths");
    }
    values.insert(values.end(), row.begin(), row.end());
  }
  return decomposition::SVD::Matrix(row_count, col_count, std::move(values));
}

std::vector<std::vector<double>>
matrix_rows(const decomposition::SVD::Matrix &matrix) {
  std::vector<std::vector<double>> rows(matrix.rows(),
                                        std::vector<double>(matrix.cols()));
  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    for (std::size_t col = 0; col < matrix.cols(); ++col) {
      rows[row][col] = matrix(row, col);
    }
  }
  return rows;
}

decomposition::SVD::PCASolver parse_pca_solver(const std::string &solver) {
  if (solver == "svd") {
    return decomposition::SVD::PCASolver::SVD;
  }
  if (solver == "covariance" || solver == "cov") {
    return decomposition::SVD::PCASolver::Covariance;
  }
  throw std::invalid_argument("solver must be 'svd' or 'covariance'");
}

mbs::numerics::InterpolationMethod parse_interpolation_method(
    const std::string &method) {
  if (method == "linear") {
    return mbs::numerics::InterpolationMethod::Linear;
  }
  if (method == "log_linear" || method == "log-linear") {
    return mbs::numerics::InterpolationMethod::LogLinear;
  }
  if (method == "flat_forward" || method == "flat-forward") {
    return mbs::numerics::InterpolationMethod::FlatForward;
  }
  if (method == "monotone_cubic" || method == "monotone-cubic") {
    return mbs::numerics::InterpolationMethod::MonotoneCubic;
  }
  throw std::invalid_argument("unsupported interpolation method");
}

std::string interpolation_method_name(
    mbs::numerics::InterpolationMethod method) {
  switch (method) {
  case mbs::numerics::InterpolationMethod::Linear:
    return "linear";
  case mbs::numerics::InterpolationMethod::LogLinear:
    return "log_linear";
  case mbs::numerics::InterpolationMethod::FlatForward:
    return "flat_forward";
  case mbs::numerics::InterpolationMethod::MonotoneCubic:
    return "monotone_cubic";
  }
  throw std::invalid_argument("unsupported interpolation method");
}

mbs::rates::RateInstrumentType parse_rate_instrument_type(
    const std::string &type) {
  if (type == "deposit") {
    return mbs::rates::RateInstrumentType::Deposit;
  }
  if (type == "zero_coupon_bond" || type == "zcb") {
    return mbs::rates::RateInstrumentType::ZeroCouponBond;
  }
  if (type == "par_swap" || type == "swap") {
    return mbs::rates::RateInstrumentType::ParSwap;
  }
  throw std::invalid_argument("unsupported rate instrument type");
}

std::string rate_instrument_type_name(mbs::rates::RateInstrumentType type) {
  switch (type) {
  case mbs::rates::RateInstrumentType::Deposit:
    return "deposit";
  case mbs::rates::RateInstrumentType::ZeroCouponBond:
    return "zero_coupon_bond";
  case mbs::rates::RateInstrumentType::ParSwap:
    return "par_swap";
  }
  throw std::invalid_argument("unsupported rate instrument type");
}

py::dict factor_key_dict(const mbs::factors::FactorKey &key) {
  py::dict result;
  result["type"] = key.type;
  result["name"] = key.name;
  return result;
}

py::dict attribution_result_dict(const mbs::attribution::AttributionResult &result) {
  py::dict out;
  out["run_id"] = result.run_id;
  out["security_id"] = result.security_id;
  out["component"] = result.component;
  out["method"] = result.method;
  out["label"] = result.label;
  out["factor_1"] =
      result.factor_1 ? py::object(factor_key_dict(*result.factor_1))
                      : py::object(py::none());
  out["factor_2"] =
      result.factor_2 ? py::object(factor_key_dict(*result.factor_2))
                      : py::object(py::none());
  out["greek"] = result.greek;
  out["bump_1"] = result.bump_1;
  out["bump_2"] = result.bump_2;
  out["realized_move_1"] = result.realized_move_1;
  out["realized_move_2"] = result.realized_move_2;
  out["pnl_price"] = result.pnl_price;
  return out;
}

py::dict pca_result_dict(const decomposition::SVD::PCAResult &result) {
  py::dict out;
  out["loadings"] = matrix_rows(result.loadings);
  out["scores"] = matrix_rows(result.scores);
  out["singular_values"] = result.singular_values;
  out["explained_variance"] = result.explained_variance;
  out["explained_variance_ratio"] = result.explained_variance_ratio;
  out["column_means"] = result.column_means;
  out["input_labels"] = result.input_labels;
  out["component_labels"] = result.component_labels;
  out["component_type"] = result.component_type;
  return out;
}

py::dict curve_node_dict(const mbs::rates::CurveNode &node) {
  py::dict out;
  out["time"] = node.time;
  out["discount"] = node.discount;
  return out;
}

py::list curve_nodes_list(const mbs::rates::DiscountCurve &curve) {
  py::list out;
  for (const auto &node : curve.nodes()) {
    out.append(curve_node_dict(node));
  }
  return out;
}

} // namespace

PYBIND11_MODULE(_cpp, m) {
  m.doc() = "C++ calculators for cross_term_greeks";

  py::class_<mbs::factors::FactorKey>(m, "FactorKey")
      .def(py::init<>())
      .def(py::init<std::string, std::string>(), py::arg("type"), py::arg("name"))
      .def_readwrite("type", &mbs::factors::FactorKey::type)
      .def_readwrite("name", &mbs::factors::FactorKey::name);

  py::class_<mbs::factors::FactorMove>(m, "FactorMove")
      .def(py::init<>())
      .def_readwrite("key", &mbs::factors::FactorMove::key)
      .def_readwrite("bump", &mbs::factors::FactorMove::bump)
      .def_readwrite("realized_move", &mbs::factors::FactorMove::realized_move);

  py::class_<mbs::factors::FactorBasis>(m, "FactorBasis")
      .def(py::init<>())
      .def_readwrite("name", &mbs::factors::FactorBasis::name)
      .def_readwrite("input_factors", &mbs::factors::FactorBasis::input_factors)
      .def_readwrite("factors", &mbs::factors::FactorBasis::factors)
      .def_readwrite("loadings", &mbs::factors::FactorBasis::loadings);

  py::class_<mbs::scenarios::ScenarioPrice>(m, "ScenarioPrice")
      .def(py::init<>())
      .def_readwrite("run_id", &mbs::scenarios::ScenarioPrice::run_id)
      .def_readwrite("security_id", &mbs::scenarios::ScenarioPrice::security_id)
      .def_readwrite("asof_date", &mbs::scenarios::ScenarioPrice::asof_date)
      .def_readwrite("scenario_id", &mbs::scenarios::ScenarioPrice::scenario_id)
      .def_readwrite("factor_type_1",
                     &mbs::scenarios::ScenarioPrice::factor_type_1)
      .def_readwrite("factor_name_1",
                     &mbs::scenarios::ScenarioPrice::factor_name_1)
      .def_readwrite("shock_1", &mbs::scenarios::ScenarioPrice::shock_1)
      .def_readwrite("factor_type_2",
                     &mbs::scenarios::ScenarioPrice::factor_type_2)
      .def_readwrite("factor_name_2",
                     &mbs::scenarios::ScenarioPrice::factor_name_2)
      .def_readwrite("shock_2", &mbs::scenarios::ScenarioPrice::shock_2)
      .def_readwrite("price", &mbs::scenarios::ScenarioPrice::price);

  py::class_<mbs::scenarios::ScenarioLattice>(m, "ScenarioLattice")
      .def(py::init<>())
      .def(py::init<std::vector<mbs::scenarios::ScenarioPrice>>(),
           py::arg("prices"))
      .def("add", &mbs::scenarios::ScenarioLattice::add, py::arg("price"))
      .def("base_price", &mbs::scenarios::ScenarioLattice::base_price,
           py::arg("run_id"), py::arg("security_id"))
      .def("shocked_price", &mbs::scenarios::ScenarioLattice::shocked_price,
           py::arg("run_id"), py::arg("security_id"), py::arg("factor"),
           py::arg("shock"))
      .def("cross_price", &mbs::scenarios::ScenarioLattice::cross_price,
           py::arg("run_id"), py::arg("security_id"), py::arg("factor_1"),
           py::arg("shock_1"), py::arg("factor_2"), py::arg("shock_2"));

  py::class_<mbs::attribution::AttributionRequest>(m, "AttributionRequest")
      .def(py::init<>())
      .def_readwrite("run_id", &mbs::attribution::AttributionRequest::run_id)
      .def_readwrite("security_id",
                     &mbs::attribution::AttributionRequest::security_id)
      .def_readwrite("component", &mbs::attribution::AttributionRequest::component)
      .def_readwrite("method", &mbs::attribution::AttributionRequest::method)
      .def_readwrite("move_1", &mbs::attribution::AttributionRequest::move_1)
      .def_readwrite("move_2", &mbs::attribution::AttributionRequest::move_2)
      .def_readwrite("moves", &mbs::attribution::AttributionRequest::moves);

  py::class_<mbs::attribution::AttributionResult>(m, "AttributionResult")
      .def(py::init<>())
      .def_readwrite("run_id", &mbs::attribution::AttributionResult::run_id)
      .def_readwrite("security_id",
                     &mbs::attribution::AttributionResult::security_id)
      .def_readwrite("component", &mbs::attribution::AttributionResult::component)
      .def_readwrite("method", &mbs::attribution::AttributionResult::method)
      .def_readwrite("label", &mbs::attribution::AttributionResult::label)
      .def_readwrite("factor_1", &mbs::attribution::AttributionResult::factor_1)
      .def_readwrite("factor_2", &mbs::attribution::AttributionResult::factor_2)
      .def_readwrite("greek", &mbs::attribution::AttributionResult::greek)
      .def_readwrite("bump_1", &mbs::attribution::AttributionResult::bump_1)
      .def_readwrite("bump_2", &mbs::attribution::AttributionResult::bump_2)
      .def_readwrite("realized_move_1",
                     &mbs::attribution::AttributionResult::realized_move_1)
      .def_readwrite("realized_move_2",
                     &mbs::attribution::AttributionResult::realized_move_2)
      .def_readwrite("pnl_price", &mbs::attribution::AttributionResult::pnl_price)
      .def("as_dict", &attribution_result_dict);

  py::class_<mbs::attribution::ScalarSecondOrderCalculator>(
      m, "ScalarSecondOrderCalculator")
      .def(py::init<>())
      .def("compute", &mbs::attribution::ScalarSecondOrderCalculator::compute,
           py::arg("lattice"), py::arg("request"));

  py::class_<mbs::attribution::CrossSecondOrderCalculator>(
      m, "CrossSecondOrderCalculator")
      .def(py::init<>())
      .def("compute", &mbs::attribution::CrossSecondOrderCalculator::compute,
           py::arg("lattice"), py::arg("request"));

  py::class_<mbs::attribution::MatrixHessianCalculator>(
      m, "MatrixHessianCalculator")
      .def(py::init<>())
      .def("compute", &mbs::attribution::MatrixHessianCalculator::compute,
           py::arg("lattice"), py::arg("request"))
      .def("compute_components",
           &mbs::attribution::MatrixHessianCalculator::compute_components,
           py::arg("lattice"), py::arg("request"));

  py::class_<mbs::numerics::Interpolator>(m, "Interpolator")
      .def(py::init([](std::vector<double> x, std::vector<double> y,
                       const std::string &method, bool allow_extrapolation) {
             return mbs::numerics::Interpolator(
                 std::move(x), std::move(y), parse_interpolation_method(method),
                 allow_extrapolation);
           }),
           py::arg("x"), py::arg("y"), py::arg("method") = "linear",
           py::arg("allow_extrapolation") = false)
      .def("value", &mbs::numerics::Interpolator::value, py::arg("x_value"))
      .def_property_readonly("x", &mbs::numerics::Interpolator::x)
      .def_property_readonly("y", &mbs::numerics::Interpolator::y)
      .def_property_readonly("method", [](const mbs::numerics::Interpolator &self) {
        return interpolation_method_name(self.method());
      });

  py::class_<mbs::rates::CurveNode>(m, "CurveNode")
      .def(py::init<>())
      .def(py::init<double, double>(), py::arg("time"), py::arg("discount"))
      .def_readwrite("time", &mbs::rates::CurveNode::time)
      .def_readwrite("discount", &mbs::rates::CurveNode::discount);

  py::class_<mbs::rates::DiscountCurve>(m, "DiscountCurve")
      .def(py::init([](std::vector<mbs::rates::CurveNode> nodes,
                       const std::string &method, bool allow_extrapolation) {
             return mbs::rates::DiscountCurve(
                 std::move(nodes), parse_interpolation_method(method),
                 allow_extrapolation);
           }),
           py::arg("nodes"), py::arg("method") = "log_linear",
           py::arg("allow_extrapolation") = false)
      .def_static("from_zero_rates",
                  [](const std::vector<double> &times,
                     const std::vector<double> &zero_rates,
                     const std::string &method) {
                    return mbs::rates::DiscountCurve::from_zero_rates(
                        times, zero_rates, parse_interpolation_method(method));
                  },
                  py::arg("times"), py::arg("zero_rates"),
                  py::arg("method") = "log_linear")
      .def("discount", &mbs::rates::DiscountCurve::discount, py::arg("time"))
      .def("zero_rate", &mbs::rates::DiscountCurve::zero_rate, py::arg("time"))
      .def("forward_rate", &mbs::rates::DiscountCurve::forward_rate,
           py::arg("start_time"), py::arg("end_time"))
      .def_property_readonly("nodes", &curve_nodes_list);

  py::class_<mbs::rates::RateInstrument>(m, "RateInstrument")
      .def(py::init([](const std::string &type, double maturity, double quote,
                       double fixed_frequency) {
             mbs::rates::RateInstrument instrument;
             instrument.type = parse_rate_instrument_type(type);
             instrument.maturity = maturity;
             instrument.quote = quote;
             instrument.fixed_frequency = fixed_frequency;
             return instrument;
           }),
           py::arg("type") = "zero_coupon_bond", py::arg("maturity") = 0.0,
           py::arg("quote") = 0.0, py::arg("fixed_frequency") = 1.0)
      .def_property(
          "type",
          [](const mbs::rates::RateInstrument &self) {
            return rate_instrument_type_name(self.type);
          },
          [](mbs::rates::RateInstrument &self, const std::string &type) {
            self.type = parse_rate_instrument_type(type);
          })
      .def_readwrite("maturity", &mbs::rates::RateInstrument::maturity)
      .def_readwrite("quote", &mbs::rates::RateInstrument::quote)
      .def_readwrite("fixed_frequency",
                     &mbs::rates::RateInstrument::fixed_frequency);

  py::class_<mbs::rates::CalibrationError>(m, "CalibrationError")
      .def_readwrite("maturity", &mbs::rates::CalibrationError::maturity)
      .def_readwrite("market_quote", &mbs::rates::CalibrationError::market_quote)
      .def_readwrite("model_quote", &mbs::rates::CalibrationError::model_quote)
      .def_readwrite("error", &mbs::rates::CalibrationError::error);

  py::class_<mbs::rates::CurveCalibrationResult>(m, "CurveCalibrationResult")
      .def_readwrite("curve", &mbs::rates::CurveCalibrationResult::curve)
      .def_readwrite("errors", &mbs::rates::CurveCalibrationResult::errors)
      .def_readwrite("rmse", &mbs::rates::CurveCalibrationResult::rmse);

  py::class_<mbs::rates::SwaptionProxyInput>(m, "SwaptionProxyInput")
      .def(py::init<>())
      .def_readwrite("expiry", &mbs::rates::SwaptionProxyInput::expiry)
      .def_readwrite("payment_times",
                     &mbs::rates::SwaptionProxyInput::payment_times)
      .def_readwrite("accruals", &mbs::rates::SwaptionProxyInput::accruals)
      .def_readwrite("forward_rate_vols",
                     &mbs::rates::SwaptionProxyInput::forward_rate_vols)
      .def_readwrite("correlation", &mbs::rates::SwaptionProxyInput::correlation);

  py::class_<mbs::rates::SwaptionProxyResult>(m, "SwaptionProxyResult")
      .def_readwrite("forward_swap_rate",
                     &mbs::rates::SwaptionProxyResult::forward_swap_rate)
      .def_readwrite("annuity", &mbs::rates::SwaptionProxyResult::annuity)
      .def_readwrite("black_vol", &mbs::rates::SwaptionProxyResult::black_vol)
      .def_readwrite("weights", &mbs::rates::SwaptionProxyResult::weights);

  py::class_<mbs::rates::LmmSofrInstrument>(m, "LmmSofrInstrument")
      .def(py::init<>())
      .def_readwrite("expiry", &mbs::rates::LmmSofrInstrument::expiry)
      .def_readwrite("tenor", &mbs::rates::LmmSofrInstrument::tenor)
      .def_readwrite("market_vol", &mbs::rates::LmmSofrInstrument::market_vol)
      .def_readwrite("weight", &mbs::rates::LmmSofrInstrument::weight);

  py::class_<mbs::rates::LmmSofrCalibrationInput>(m, "LmmSofrCalibrationInput")
      .def(py::init<>())
      .def_readwrite("forward_start_times",
                     &mbs::rates::LmmSofrCalibrationInput::forward_start_times)
      .def_readwrite("accruals", &mbs::rates::LmmSofrCalibrationInput::accruals)
      .def_readwrite("initial_vols",
                     &mbs::rates::LmmSofrCalibrationInput::initial_vols)
      .def_readwrite("correlation",
                     &mbs::rates::LmmSofrCalibrationInput::correlation)
      .def_readwrite("instruments",
                     &mbs::rates::LmmSofrCalibrationInput::instruments);

  py::class_<mbs::rates::LmmSofrCalibrationResult>(
      m, "LmmSofrCalibrationResult")
      .def_readwrite("forward_vols",
                     &mbs::rates::LmmSofrCalibrationResult::forward_vols)
      .def_readwrite("model_vols",
                     &mbs::rates::LmmSofrCalibrationResult::model_vols)
      .def_readwrite("weighted_rmse",
                     &mbs::rates::LmmSofrCalibrationResult::weighted_rmse);

  m.def("central_second", &mbs::numerics::central_second, py::arg("p_base"),
        py::arg("p_up"), py::arg("p_down"), py::arg("p_bump"));
  m.def("central_cross", &mbs::numerics::central_cross, py::arg("p_up_up"),
        py::arg("p_up_down"), py::arg("p_down_up"), py::arg("p_down_down"),
        py::arg("bump_1"), py::arg("bump_2"));
  m.def("second_order_pnl", &mbs::numerics::second_order_pnl, py::arg("gamma"),
        py::arg("realized_move"));
  m.def("cross_pnl", &mbs::numerics::cross_pnl, py::arg("cross_greek"),
        py::arg("realized_move_1"), py::arg("realized_move_2"));

  m.def(
      "fit_pca",
      [](const std::vector<std::vector<double>> &observations,
         std::size_t n_components, const std::vector<std::string> &input_labels,
         const std::string &component_type, const std::string &component_prefix,
         bool center, const std::string &solver) {
        decomposition::SVD::PCAOptions options;
        options.center_columns = center;
        options.components = n_components;
        options.solver = parse_pca_solver(solver);
        options.factor_type = component_type;
        options.factor_name_prefix = component_prefix;
        const decomposition::SVD::SVDInput input(
            make_matrix(observations), {}, input_labels);
        const auto result =
            decomposition::SVD::PCATransformer(options).fit_transform(input);
        return pca_result_dict(result);
      },
      py::arg("observations"), py::arg("n_components") = 0,
      py::arg("input_labels") = std::vector<std::string>{},
      py::arg("component_type") = "PCA", py::arg("component_prefix") = "PC",
      py::arg("center") = true, py::arg("solver") = "svd");

  m.def("bootstrap_discount_curve",
        [](const std::vector<mbs::rates::RateInstrument> &instruments,
           const std::string &method) {
          return mbs::rates::bootstrap_discount_curve(
              instruments, parse_interpolation_method(method));
        },
        py::arg("instruments"), py::arg("method") = "log_linear");
  m.def("par_swap_rate", &mbs::rates::par_swap_rate, py::arg("curve"),
        py::arg("maturity"), py::arg("fixed_frequency") = 1.0);
  m.def("calibrate_bootstrap_curve",
        [](const std::vector<mbs::rates::RateInstrument> &instruments,
           const std::string &method) {
          return mbs::rates::calibrate_bootstrap_curve(
              instruments, parse_interpolation_method(method));
        },
        py::arg("instruments"), py::arg("method") = "log_linear");
  m.def("derive_swaption_proxy_vol", &mbs::rates::derive_swaption_proxy_vol,
        py::arg("curve"), py::arg("input"));
  m.def("calibrate_lmm_sofr_proxy", &mbs::rates::calibrate_lmm_sofr_proxy,
        py::arg("curve"), py::arg("input"), py::arg("max_iterations") = 32,
        py::arg("tolerance") = 1.0e-10);
}
