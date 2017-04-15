#include <sstream>
#include <fstream>
#include <iostream>

#include "RelationalAlgebra.pb.h"

namespace common = blox::common::protocol;
namespace lb = blox::compiler::gpu;

int main()
{
        std::ofstream raFile("logicQ1.man.binaryGpu");
        std::ofstream outTXTFile("logicQ1.man.textGpu");

        if(!raFile.is_open())
        {
                std::cout << "Failed to open RA IR " <<
                        "protocol buffer file: 'logicQ1.man.binaryGpu' for writing.\n";
				exit(-1);
        }

		if(!outTXTFile.is_open())
        {
                std::cout << "Failed to open RA IR protocol buffer file for writing text.\n";
				exit(-1);
        }


        lb::GPUGraph raGraph;

	raGraph.set_graphname("TPC-H/logicQ1.lbb");

	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("+$logicQ1:_dateDelta");

		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_INT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("+$logicQ1:_sum_qty");

		{
			common::Type& raField = *raVariable.add_keys();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}
		{
			common::Type& raField = *raVariable.add_keys();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("LINESTATUS");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("+$logicQ1:_sum_base_price");

		{
			common::Type& raField = *raVariable.add_keys();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}
		{
			common::Type& raField = *raVariable.add_keys();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("LINESTATUS");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("+$logicQ1:_avg_disc_adt");

		{
			common::Type& raField = *raVariable.add_keys();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}
		{
			common::Type& raField = *raVariable.add_keys();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("LINESTATUS");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("+$logicQ1:_sum_disc_price");

		{
			common::Type& raField = *raVariable.add_keys();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}
		{
			common::Type& raField = *raVariable.add_keys();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("LINESTATUS");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("+$logicQ1:_sum_charge");

		{
			common::Type& raField = *raVariable.add_keys();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}
		{
			common::Type& raField = *raVariable.add_keys();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("LINESTATUS");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("+$logicQ1:_count_order");

		{
			common::Type& raField = *raVariable.add_keys();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}
		{
			common::Type& raField = *raVariable.add_keys();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("LINESTATUS");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_INT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("L_SHIPDATE");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_DATETIME);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("L_RETURNFLAG");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("L_LINESTATUS");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("LINESTATUS");
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("L_QUANTITY");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("L_EXTENDEDPRICE");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("L_DISCOUNT");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("L_TAX");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("+$logicQ1:_startDate");

		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_DATETIME);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("j_1");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_DATETIME);
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_DATETIME);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("m_1");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("j_2");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("j_3");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("LINESTATUS");
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("j_4");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("LINESTATUS");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("j_5");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("LINESTATUS");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("j_6");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("LINESTATUS");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("j_7");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("LINESTATUS");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("m_2");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("LINESTATUS");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("m_3");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("LINESTATUS");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("m_4");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("LINESTATUS");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUVariable& raVariable = *raGraph.add_variables();
		raVariable.set_varname("m_5");

		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("LINENUMBER");
		}
		{
			common::Type& raKey = *raVariable.add_keys();
			raKey.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raKey.mutable_unary();
			unary.set_name("ORDER");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("RETURNFLAG");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_UNARY);

			common::UnaryPredicateType& unary = *raField.mutable_unary();
			unary.set_name("LINESTATUS");
		}
		{
			common::Type& raField = *raVariable.add_fields();
			raField.set_kind(common::Type_Kind_PRIMITIVE);

			common::PrimitiveType& primitive = *raField.mutable_primitive();
			primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
			primitive.set_capacity(64);
		}

		raVariable.set_initialdata("");
	}
	{
	        lb::GPUSequence& raSequence = *raGraph.add_sequences();
		raSequence.set_uniqueidentifier(0);

		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("+$logicQ1:_startDate");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::MAPFILTER);

			lb::GPUMapFilter& mapfilter = *op.mutable_mapfilter();
			mapfilter.set_srcwidth(1);
			mapfilter.set_srca("+$logicQ1:_dateDelta");

			lb::Compare& compare = *mapfilter.mutable_predicate();
			compare.set_tag(lb::ALWAYS);

			{
				lb::Exp& exp = *mapfilter.add_reordering();
				exp.set_tag(lb::CALL);
				
				lb::Call& call = *exp.mutable_call();

				call.set_calltype(lb::FDatetime);
				call.set_callname("subtract");

				{
					lb::Exp& arg1 = *call.add_args();
					arg1.set_tag(lb::CONSTEXP);

					lb::ConstExp& constExp = *arg1.mutable_constexp();
					common::Constant& constant = *constExp.mutable_literal();
					constant.set_kind(common::Constant_Kind_DATETIME);
					common::DateTimeConstant& datetime_constant = *constant.mutable_date_time_constant();
					datetime_constant.set_value(912470400);
				}
				{
					lb::Exp& arg2 = *call.add_args();
					arg2.set_tag(lb::INDEX);

					lb::Index& index = *arg2.mutable_index();
					index.set_offset(0);

					common::Type& etyp = *index.mutable_etyp();
					etyp.set_kind(common::Type_Kind_PRIMITIVE);
		
					common::PrimitiveType& primitive = *etyp.mutable_primitive();
					primitive.set_kind(common::PrimitiveType_Kind_INT);
					primitive.set_capacity(64);
				}
				{
					lb::Exp& arg3 = *call.add_args();
					arg3.set_tag(lb::CONSTEXP);

					lb::ConstExp& constExp = *arg3.mutable_constexp();
					common::Constant& constant = *constExp.mutable_literal();
					constant.set_kind(common::Constant_Kind_STRING);
					common::StringConstant& string_constant = *constant.mutable_string_constant();
					string_constant.set_value("days");
				}
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("j_1");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::JOIN);

			lb::GPUJoin& join = *op.mutable_join();
			join.set_keycount(0);
			join.set_srca("L_SHIPDATE");
			join.set_srcb("+$logicQ1:_startDate");

			{
				lb::Exp& arg1 = *join.add_args();
				arg1.set_tag(lb::INDEX);

				lb::Index& index = *arg1.mutable_index();
				index.set_offset(0);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINENUMBER");
			}
			{
				lb::Exp& arg2 = *join.add_args();
				arg2.set_tag(lb::INDEX);

				lb::Index& index = *arg2.mutable_index();
				index.set_offset(1);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("ORDER");
			}
			{
				lb::Exp& arg3 = *join.add_args();
				arg3.set_tag(lb::INDEX);

				lb::Index& index = *arg3.mutable_index();
				index.set_offset(2);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_PRIMITIVE);
		
				common::PrimitiveType& primitive = *etyp.mutable_primitive();
				primitive.set_kind(common::PrimitiveType_Kind_DATETIME);
				primitive.set_capacity(64);
			}
			{
				lb::Exp& arg4 = *join.add_args();
				arg4.set_tag(lb::INDEX);

				lb::Index& index = *arg4.mutable_index();
				index.set_offset(3);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_PRIMITIVE);
		
				common::PrimitiveType& primitive = *etyp.mutable_primitive();
				primitive.set_kind(common::PrimitiveType_Kind_DATETIME);
				primitive.set_capacity(64);
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("m_1");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::MAPFILTER);

			lb::GPUMapFilter& mapfilter = *op.mutable_mapfilter();
			mapfilter.set_srcwidth(4);
			mapfilter.set_srca("j_1");

			lb::Compare& compare = *mapfilter.mutable_predicate();
			compare.set_tag(lb::COMP);

			lb::Comp& comp = *compare.mutable_comp();
			comp.set_comparison(lb::Le);

			{
				lb::Exp& op1 = *comp.mutable_op1();
				op1.set_tag(lb::INDEX);

				lb::Index& index = *op1.mutable_index();
				index.set_offset(2);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_PRIMITIVE);
		
				common::PrimitiveType& primitive = *etyp.mutable_primitive();
				primitive.set_kind(common::PrimitiveType_Kind_DATETIME);
				primitive.set_capacity(64);
			}
			{
				lb::Exp& op2 = *comp.mutable_op2();
				op2.set_tag(lb::INDEX);

				lb::Index& index = *op2.mutable_index();
				index.set_offset(3);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_PRIMITIVE);
		
				common::PrimitiveType& primitive = *etyp.mutable_primitive();
				primitive.set_kind(common::PrimitiveType_Kind_DATETIME);
				primitive.set_capacity(64);
			}
			{
				lb::Exp& arg1 = *mapfilter.add_reordering();
				arg1.set_tag(lb::INDEX);

				lb::Index& index = *arg1.mutable_index();
				index.set_offset(0);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINENUMBER");
			}
			{
				lb::Exp& arg2 = *mapfilter.add_reordering();
				arg2.set_tag(lb::INDEX);

				lb::Index& index = *arg2.mutable_index();
				index.set_offset(1);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("ORDER");
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("j_2");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::JOIN);

			lb::GPUJoin& join = *op.mutable_join();
			join.set_keycount(2);
			join.set_srca("m_1");
			join.set_srcb("L_RETURNFLAG");

			{
				lb::Exp& arg1 = *join.add_args();
				arg1.set_tag(lb::INDEX);

				lb::Index& index = *arg1.mutable_index();
				index.set_offset(0);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINENUMBER");
			}
			{
				lb::Exp& arg2 = *join.add_args();
				arg2.set_tag(lb::INDEX);

				lb::Index& index = *arg2.mutable_index();
				index.set_offset(1);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("ORDER");
			}
			{
				lb::Exp& arg3 = *join.add_args();
				arg3.set_tag(lb::INDEX);

				lb::Index& index = *arg3.mutable_index();
				index.set_offset(2);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("RETURNFLAG");
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("j_3");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::JOIN);

			lb::GPUJoin& join = *op.mutable_join();
			join.set_keycount(2);
			join.set_srca("j_2");
			join.set_srcb("L_LINESTATUS");

			{
				lb::Exp& arg1 = *join.add_args();
				arg1.set_tag(lb::INDEX);

				lb::Index& index = *arg1.mutable_index();
				index.set_offset(0);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINENUMBER");
			}
			{
				lb::Exp& arg2 = *join.add_args();
				arg2.set_tag(lb::INDEX);

				lb::Index& index = *arg2.mutable_index();
				index.set_offset(1);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("ORDER");
			}
			{
				lb::Exp& arg3 = *join.add_args();
				arg3.set_tag(lb::INDEX);

				lb::Index& index = *arg3.mutable_index();
				index.set_offset(2);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("RETURNFLAG");
			}
			{
				lb::Exp& arg4 = *join.add_args();
				arg4.set_tag(lb::INDEX);

				lb::Index& index = *arg4.mutable_index();
				index.set_offset(3);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINESTATUS");
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("+$logicQ1:_count_order");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::AGG);

			lb::GPUAgg& agg = *op.mutable_agg();
			agg.set_srcwidth(4);
			agg.set_srca("j_3");
			agg.add_domains(2);
			agg.add_domains(3);

			{
				lb::Agg& range = *agg.add_range();
				range.set_tag(lb::COUNT);

				lb::Count& count = *range.mutable_count();
				count.set_aggrng(0);
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("j_4");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::JOIN);

			lb::GPUJoin& join = *op.mutable_join();
			join.set_keycount(2);
			join.set_srca("j_3");
			join.set_srcb("L_QUANTITY");

			{
				lb::Exp& arg1 = *join.add_args();
				arg1.set_tag(lb::INDEX);

				lb::Index& index = *arg1.mutable_index();
				index.set_offset(0);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINENUMBER");
			}
			{
				lb::Exp& arg2 = *join.add_args();
				arg2.set_tag(lb::INDEX);

				lb::Index& index = *arg2.mutable_index();
				index.set_offset(1);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("ORDER");
			}
			{
				lb::Exp& arg3 = *join.add_args();
				arg3.set_tag(lb::INDEX);

				lb::Index& index = *arg3.mutable_index();
				index.set_offset(2);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("RETURNFLAG");
			}
			{
				lb::Exp& arg4 = *join.add_args();
				arg4.set_tag(lb::INDEX);

				lb::Index& index = *arg4.mutable_index();
				index.set_offset(3);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINESTATUS");
			}
			{
				lb::Exp& arg5 = *join.add_args();
				arg5.set_tag(lb::INDEX);

				lb::Index& index = *arg5.mutable_index();
				index.set_offset(4);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_PRIMITIVE);

				common::PrimitiveType& primitive = *etyp.mutable_primitive();
				primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
				primitive.set_capacity(64);
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("+$logicQ1:_sum_qty");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::AGG);

			lb::GPUAgg& agg = *op.mutable_agg();
			agg.set_srcwidth(5);
			agg.set_srca("j_4");
			agg.add_domains(2);
			agg.add_domains(3);

			{
				lb::Agg& range = *agg.add_range();
				range.set_tag(lb::TOTAL);

				lb::Total& total = *range.mutable_total();
				total.set_aggdom(4);
				total.set_aggrng(4);
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("j_5");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::JOIN);

			lb::GPUJoin& join = *op.mutable_join();
			join.set_keycount(2);
			join.set_srca("j_3");
			join.set_srcb("L_EXTENDEDPRICE");

			{
				lb::Exp& arg1 = *join.add_args();
				arg1.set_tag(lb::INDEX);

				lb::Index& index = *arg1.mutable_index();
				index.set_offset(0);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINENUMBER");
			}
			{
				lb::Exp& arg2 = *join.add_args();
				arg2.set_tag(lb::INDEX);

				lb::Index& index = *arg2.mutable_index();
				index.set_offset(1);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("ORDER");
			}
			{
				lb::Exp& arg3 = *join.add_args();
				arg3.set_tag(lb::INDEX);

				lb::Index& index = *arg3.mutable_index();
				index.set_offset(2);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("RETURNFLAG");
			}
			{
				lb::Exp& arg4 = *join.add_args();
				arg4.set_tag(lb::INDEX);

				lb::Index& index = *arg4.mutable_index();
				index.set_offset(3);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINESTATUS");
			}
			{
				lb::Exp& arg5 = *join.add_args();
				arg5.set_tag(lb::INDEX);

				lb::Index& index = *arg5.mutable_index();
				index.set_offset(4);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_PRIMITIVE);

				common::PrimitiveType& primitive = *etyp.mutable_primitive();
				primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
				primitive.set_capacity(64);
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("+$logicQ1:_sum_base_price");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::AGG);

			lb::GPUAgg& agg = *op.mutable_agg();
			agg.set_srcwidth(5);
			agg.set_srca("j_5");
			agg.add_domains(2);
			agg.add_domains(3);

			{
				lb::Agg& range = *agg.add_range();
				range.set_tag(lb::TOTAL);

				lb::Total& total = *range.mutable_total();
				total.set_aggdom(4);
				total.set_aggrng(4);
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("j_6");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::JOIN);

			lb::GPUJoin& join = *op.mutable_join();
			join.set_keycount(2);
			join.set_srca("j_5");
			join.set_srcb("L_DISCOUNT");

			{
				lb::Exp& arg1 = *join.add_args();
				arg1.set_tag(lb::INDEX);

				lb::Index& index = *arg1.mutable_index();
				index.set_offset(0);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINENUMBER");
			}
			{
				lb::Exp& arg2 = *join.add_args();
				arg2.set_tag(lb::INDEX);

				lb::Index& index = *arg2.mutable_index();
				index.set_offset(1);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("ORDER");
			}
			{
				lb::Exp& arg3 = *join.add_args();
				arg3.set_tag(lb::INDEX);

				lb::Index& index = *arg3.mutable_index();
				index.set_offset(2);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("RETURNFLAG");
			}
			{
				lb::Exp& arg4 = *join.add_args();
				arg4.set_tag(lb::INDEX);

				lb::Index& index = *arg4.mutable_index();
				index.set_offset(3);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINESTATUS");
			}
			{
				lb::Exp& arg5 = *join.add_args();
				arg5.set_tag(lb::INDEX);

				lb::Index& index = *arg5.mutable_index();
				index.set_offset(4);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_PRIMITIVE);

				common::PrimitiveType& primitive = *etyp.mutable_primitive();
				primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
				primitive.set_capacity(64);
			}
			{
				lb::Exp& arg6 = *join.add_args();
				arg6.set_tag(lb::INDEX);

				lb::Index& index = *arg6.mutable_index();
				index.set_offset(5);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_PRIMITIVE);

				common::PrimitiveType& primitive = *etyp.mutable_primitive();
				primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
				primitive.set_capacity(64);
			}

		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("+$logicQ1:_avg_disc_adt");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::AGG);

			lb::GPUAgg& agg = *op.mutable_agg();
			agg.set_srcwidth(6);
			agg.set_srca("j_6");
			agg.add_domains(2);
			agg.add_domains(3);

			{
				lb::Agg& range = *agg.add_range();
				range.set_tag(lb::TOTAL);

				lb::Total& total = *range.mutable_total();
				total.set_aggdom(5);
				total.set_aggrng(5);
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("m_2");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::MAPFILTER);

			lb::GPUMapFilter& mapfilter = *op.mutable_mapfilter();
			mapfilter.set_srcwidth(6);
			mapfilter.set_srca("j_6");

			lb::Compare& compare = *mapfilter.mutable_predicate();
			compare.set_tag(lb::ALWAYS);

			{
				lb::Exp& arg1 = *mapfilter.add_reordering();
				arg1.set_tag(lb::INDEX);

				lb::Index& index = *arg1.mutable_index();
				index.set_offset(0);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINENUMBER");
			}
			{
				lb::Exp& arg2 = *mapfilter.add_reordering();
				arg2.set_tag(lb::INDEX);

				lb::Index& index = *arg2.mutable_index();
				index.set_offset(1);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("ORDER");
			}
			{
				lb::Exp& arg3 = *mapfilter.add_reordering();
				arg3.set_tag(lb::INDEX);

				lb::Index& index = *arg3.mutable_index();
				index.set_offset(2);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("RETURNFLAG");
			}
			{
				lb::Exp& arg4 = *mapfilter.add_reordering();
				arg4.set_tag(lb::INDEX);

				lb::Index& index = *arg4.mutable_index();
				index.set_offset(3);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINESTATUS");
			}
			{
				lb::Exp& arg5 = *mapfilter.add_reordering();
				arg5.set_tag(lb::INDEX);

				lb::Index& index = *arg5.mutable_index();
				index.set_offset(4);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_PRIMITIVE);

				common::PrimitiveType& primitive = *etyp.mutable_primitive();
				primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
				primitive.set_capacity(64);
			}
			{
				lb::Exp& arg6 = *mapfilter.add_reordering();
				arg6.set_tag(lb::INDEX);

				lb::Index& index = *arg6.mutable_index();
				index.set_offset(5);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_PRIMITIVE);

				common::PrimitiveType& primitive = *etyp.mutable_primitive();
				primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
				primitive.set_capacity(64);
			}
			{
				lb::Exp& exp = *mapfilter.add_reordering();
				exp.set_tag(lb::ARITHEXP);
				
				lb::ArithExp& arithexp = *exp.mutable_arithexp();
				lb::NumType& domain = *arithexp.mutable_domain();
				domain.set_typ(lb::NFloat);
				domain.set_size(64);
				arithexp.set_op(lb::Multiply);

				{
					lb::Exp& exp1 = *arithexp.mutable_exp1();
					exp1.set_tag(lb::INDEX);

					lb::Index& index = *exp1.mutable_index();
					index.set_offset(4);

					common::Type& etyp = *index.mutable_etyp();
					etyp.set_kind(common::Type_Kind_PRIMITIVE);
		
					common::PrimitiveType& primitive = *etyp.mutable_primitive();
					primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
					primitive.set_capacity(64);
				}
				{
					lb::Exp& exp2 = *arithexp.mutable_exp2();
					exp2.set_tag(lb::ARITHEXP);
					
					lb::ArithExp& sub_arithexp = *exp2.mutable_arithexp();
					lb::NumType& sub_domain = *sub_arithexp.mutable_domain();
					sub_domain.set_typ(lb::NFloat);
					sub_domain.set_size(64);
					sub_arithexp.set_op(lb::Subtract);

					{
						lb::Exp& sub_exp1 = *sub_arithexp.mutable_exp1();
						sub_exp1.set_tag(lb::CONSTEXP);
	
						lb::ConstExp& constExp = *sub_exp1.mutable_constexp();
						common::Constant& constant = *constExp.mutable_literal();
						constant.set_kind(common::Constant_Kind_FLOAT);
						common::FloatConstant& float_constant = *constant.mutable_float_constant();
						float_constant.set_value("1");

						common::Type& type = *float_constant.mutable_type();
						type.set_kind(common::Type_Kind_PRIMITIVE);
			
						common::PrimitiveType& primitive = *type.mutable_primitive();
						primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
						primitive.set_capacity(64);
					}
					{
						lb::Exp& sub_exp2 = *sub_arithexp.mutable_exp2();
						sub_exp2.set_tag(lb::INDEX);
	
						lb::Index& index = *sub_exp2.mutable_index();
						index.set_offset(5);
	
						common::Type& etyp = *index.mutable_etyp();
						etyp.set_kind(common::Type_Kind_PRIMITIVE);
			
						common::PrimitiveType& primitive = *etyp.mutable_primitive();
						primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
						primitive.set_capacity(64);
					}
				}
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("m_3");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::MAPFILTER);

			lb::GPUMapFilter& mapfilter = *op.mutable_mapfilter();
			mapfilter.set_srcwidth(7);
			mapfilter.set_srca("m_2");

			lb::Compare& compare = *mapfilter.mutable_predicate();
			compare.set_tag(lb::ALWAYS);

			{
				lb::Exp& arg1 = *mapfilter.add_reordering();
				arg1.set_tag(lb::INDEX);

				lb::Index& index = *arg1.mutable_index();
				index.set_offset(0);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINENUMBER");
			}
			{
				lb::Exp& arg2 = *mapfilter.add_reordering();
				arg2.set_tag(lb::INDEX);

				lb::Index& index = *arg2.mutable_index();
				index.set_offset(1);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("ORDER");
			}
			{
				lb::Exp& arg3 = *mapfilter.add_reordering();
				arg3.set_tag(lb::INDEX);

				lb::Index& index = *arg3.mutable_index();
				index.set_offset(2);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("RETURNFLAG");
			}
			{
				lb::Exp& arg4 = *mapfilter.add_reordering();
				arg4.set_tag(lb::INDEX);

				lb::Index& index = *arg4.mutable_index();
				index.set_offset(3);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINESTATUS");
			}
			{
				lb::Exp& arg5 = *mapfilter.add_reordering();
				arg5.set_tag(lb::INDEX);

				lb::Index& index = *arg5.mutable_index();
				index.set_offset(6);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_PRIMITIVE);

				common::PrimitiveType& primitive = *etyp.mutable_primitive();
				primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
				primitive.set_capacity(64);
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("+$logicQ1:_sum_disc_price");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::AGG);

			lb::GPUAgg& agg = *op.mutable_agg();
			agg.set_srcwidth(5);
			agg.set_srca("m_3");
			agg.add_domains(2);
			agg.add_domains(3);

			{
				lb::Agg& range = *agg.add_range();
				range.set_tag(lb::TOTAL);

				lb::Total& total = *range.mutable_total();
				total.set_aggdom(4);
				total.set_aggrng(4);
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("j_7");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::JOIN);

			lb::GPUJoin& join = *op.mutable_join();
			join.set_keycount(2);
			join.set_srca("m_3");
			join.set_srcb("L_TAX");

			{
				lb::Exp& arg1 = *join.add_args();
				arg1.set_tag(lb::INDEX);

				lb::Index& index = *arg1.mutable_index();
				index.set_offset(0);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINENUMBER");
			}
			{
				lb::Exp& arg2 = *join.add_args();
				arg2.set_tag(lb::INDEX);

				lb::Index& index = *arg2.mutable_index();
				index.set_offset(1);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("ORDER");
			}
			{
				lb::Exp& arg3 = *join.add_args();
				arg3.set_tag(lb::INDEX);

				lb::Index& index = *arg3.mutable_index();
				index.set_offset(2);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("RETURNFLAG");
			}
			{
				lb::Exp& arg4 = *join.add_args();
				arg4.set_tag(lb::INDEX);

				lb::Index& index = *arg4.mutable_index();
				index.set_offset(3);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINESTATUS");
			}
			{
				lb::Exp& arg5 = *join.add_args();
				arg5.set_tag(lb::INDEX);

				lb::Index& index = *arg5.mutable_index();
				index.set_offset(4);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_PRIMITIVE);

				common::PrimitiveType& primitive = *etyp.mutable_primitive();
				primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
				primitive.set_capacity(64);
			}
			{
				lb::Exp& arg6 = *join.add_args();
				arg6.set_tag(lb::INDEX);

				lb::Index& index = *arg6.mutable_index();
				index.set_offset(5);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_PRIMITIVE);

				common::PrimitiveType& primitive = *etyp.mutable_primitive();
				primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
				primitive.set_capacity(64);
			}

		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("m_4");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::MAPFILTER);

			lb::GPUMapFilter& mapfilter = *op.mutable_mapfilter();
			mapfilter.set_srcwidth(6);
			mapfilter.set_srca("j_7");

			lb::Compare& compare = *mapfilter.mutable_predicate();
			compare.set_tag(lb::ALWAYS);

			{
				lb::Exp& arg1 = *mapfilter.add_reordering();
				arg1.set_tag(lb::INDEX);

				lb::Index& index = *arg1.mutable_index();
				index.set_offset(0);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINENUMBER");
			}
			{
				lb::Exp& arg2 = *mapfilter.add_reordering();
				arg2.set_tag(lb::INDEX);

				lb::Index& index = *arg2.mutable_index();
				index.set_offset(1);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("ORDER");
			}
			{
				lb::Exp& arg3 = *mapfilter.add_reordering();
				arg3.set_tag(lb::INDEX);

				lb::Index& index = *arg3.mutable_index();
				index.set_offset(2);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("RETURNFLAG");
			}
			{
				lb::Exp& arg4 = *mapfilter.add_reordering();
				arg4.set_tag(lb::INDEX);

				lb::Index& index = *arg4.mutable_index();
				index.set_offset(3);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINESTATUS");
			}
			{
				lb::Exp& arg5 = *mapfilter.add_reordering();
				arg5.set_tag(lb::INDEX);

				lb::Index& index = *arg5.mutable_index();
				index.set_offset(4);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_PRIMITIVE);

				common::PrimitiveType& primitive = *etyp.mutable_primitive();
				primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
				primitive.set_capacity(64);
			}
			{
				lb::Exp& arg6 = *mapfilter.add_reordering();
				arg6.set_tag(lb::INDEX);

				lb::Index& index = *arg6.mutable_index();
				index.set_offset(5);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_PRIMITIVE);

				common::PrimitiveType& primitive = *etyp.mutable_primitive();
				primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
				primitive.set_capacity(64);
			}
			{
				lb::Exp& exp = *mapfilter.add_reordering();
				exp.set_tag(lb::ARITHEXP);
				
				lb::ArithExp& arithexp = *exp.mutable_arithexp();
				lb::NumType& domain = *arithexp.mutable_domain();
				domain.set_typ(lb::NFloat);
				domain.set_size(64);
				arithexp.set_op(lb::Multiply);

				{
					lb::Exp& exp1 = *arithexp.mutable_exp1();
					exp1.set_tag(lb::INDEX);

					lb::Index& index = *exp1.mutable_index();
					index.set_offset(4);

					common::Type& etyp = *index.mutable_etyp();
					etyp.set_kind(common::Type_Kind_PRIMITIVE);
		
					common::PrimitiveType& primitive = *etyp.mutable_primitive();
					primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
					primitive.set_capacity(64);
				}
				{
					lb::Exp& exp2 = *arithexp.mutable_exp2();
					exp2.set_tag(lb::ARITHEXP);
					
					lb::ArithExp& sub_arithexp = *exp2.mutable_arithexp();
					lb::NumType& sub_domain = *sub_arithexp.mutable_domain();
					sub_domain.set_typ(lb::NFloat);
					sub_domain.set_size(64);
					sub_arithexp.set_op(lb::Add);

					{
						lb::Exp& sub_exp1 = *sub_arithexp.mutable_exp1();
						sub_exp1.set_tag(lb::CONSTEXP);
	
						lb::ConstExp& constExp = *sub_exp1.mutable_constexp();
						common::Constant& constant = *constExp.mutable_literal();
						constant.set_kind(common::Constant_Kind_FLOAT);
						common::FloatConstant& float_constant = *constant.mutable_float_constant();
						float_constant.set_value("1");

						common::Type& type = *float_constant.mutable_type();
						type.set_kind(common::Type_Kind_PRIMITIVE);
			
						common::PrimitiveType& primitive = *type.mutable_primitive();
						primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
						primitive.set_capacity(64);
					}
					{
						lb::Exp& sub_exp2 = *sub_arithexp.mutable_exp2();
						sub_exp2.set_tag(lb::INDEX);
	
						lb::Index& index = *sub_exp2.mutable_index();
						index.set_offset(5);
	
						common::Type& etyp = *index.mutable_etyp();
						etyp.set_kind(common::Type_Kind_PRIMITIVE);
			
						common::PrimitiveType& primitive = *etyp.mutable_primitive();
						primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
						primitive.set_capacity(64);
					}
				}
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("m_5");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::MAPFILTER);

			lb::GPUMapFilter& mapfilter = *op.mutable_mapfilter();
			mapfilter.set_srcwidth(7);
			mapfilter.set_srca("m_4");

			lb::Compare& compare = *mapfilter.mutable_predicate();
			compare.set_tag(lb::ALWAYS);

			{
				lb::Exp& arg1 = *mapfilter.add_reordering();
				arg1.set_tag(lb::INDEX);

				lb::Index& index = *arg1.mutable_index();
				index.set_offset(0);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINENUMBER");
			}
			{
				lb::Exp& arg2 = *mapfilter.add_reordering();
				arg2.set_tag(lb::INDEX);

				lb::Index& index = *arg2.mutable_index();
				index.set_offset(1);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);

				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("ORDER");
			}
			{
				lb::Exp& arg3 = *mapfilter.add_reordering();
				arg3.set_tag(lb::INDEX);

				lb::Index& index = *arg3.mutable_index();
				index.set_offset(2);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("RETURNFLAG");
			}
			{
				lb::Exp& arg4 = *mapfilter.add_reordering();
				arg4.set_tag(lb::INDEX);

				lb::Index& index = *arg4.mutable_index();
				index.set_offset(3);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_UNARY);
		
				common::UnaryPredicateType& unary = *etyp.mutable_unary();
				unary.set_name("LINESTATUS");
			}
			{
				lb::Exp& arg5 = *mapfilter.add_reordering();
				arg5.set_tag(lb::INDEX);

				lb::Index& index = *arg5.mutable_index();
				index.set_offset(6);

				common::Type& etyp = *index.mutable_etyp();
				etyp.set_kind(common::Type_Kind_PRIMITIVE);

				common::PrimitiveType& primitive = *etyp.mutable_primitive();
				primitive.set_kind(common::PrimitiveType_Kind_FLOAT);
				primitive.set_capacity(64);
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::ASSIGN);

			lb::GPUAssign& assign = *raOperator.mutable_assign();
			assign.set_dest("+$logicQ1:_sum_charge");

			lb::GPUOperator& op = *assign.mutable_op();
			op.set_tag(lb::AGG);

			lb::GPUAgg& agg = *op.mutable_agg();
			agg.set_srcwidth(5);
			agg.set_srca("m_5");
			agg.add_domains(2);
			agg.add_domains(3);

			{
				lb::Agg& range = *agg.add_range();
				range.set_tag(lb::TOTAL);

				lb::Total& total = *range.mutable_total();
				total.set_aggdom(4);
				total.set_aggrng(4);
			}
		}
		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::GOTO);

			lb::GPUGoto& jump = *raOperator.mutable_jump();
			jump.set_target(1);
		}
	}

	{
	        lb::GPUSequence& raSequence = *raGraph.add_sequences();
		raSequence.set_uniqueidentifier(1);

		{
			lb::GPUCommand& raOperator = *raSequence.add_operators();
			raOperator.set_tag(lb::HALT);
		}
	}

	raGraph.set_entry(0);
	raGraph.set_exit(1);

        std::stringstream raTempBuffer;

        if(!raGraph.SerializeToOstream(&raTempBuffer))
        {
                std::cout << "Failed to serialize protocol buffer "
                        "containing RA IR.";
		exit(-1);
        }

        long long unsigned int bytes = raTempBuffer.str().size();
        raFile.write(raTempBuffer.str().c_str(), bytes);

	std::string raText = raGraph.DebugString();

	size_t txtBytes = raText.size();
	outTXTFile.write(raText.c_str(), txtBytes);

	raFile.close();
	outTXTFile.close();

	return 0;
}
