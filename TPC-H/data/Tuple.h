/*      
        \date Tuesday November 30, 2010
        \author Gregory Diamos <gregory.diamos@gatech.edu>
        \brief The 
*/

#ifndef TUPLE_H_INCLUDED
#define TUPLE_H_INCLUDED

/*! \brief A nonexistant tuple member */
class Null {};

/*! \brief Get the min of two ints */
template<unsigned int l, unsigned int r>
class Min
{
public:
        static const unsigned int value = (l < r) ? l : r;
};

/*! \brief Clamp a value to the next highest basic unit */
template<unsigned int v>
class Clamp
{
public:
        static const unsigned int value = 
                (v <= 8) ? 8 : (v <= 16) ? 16 : (v <= 32) ? 32 : (v <= 64) ? 64 : v;
};

/*! \brief Get the next highest power of two */
template<unsigned int i>
class NextHighestPowerOfTwo
{
public:
        static const unsigned int i0 = i - 1;
        static const unsigned int i1 = i0 | (i >> 1);
        static const unsigned int i2 = i1 | (i1 >> 2);
        static const unsigned int i3 = i2 | (i2 >> 4);
        static const unsigned int i4 = i3 | (i3 >> 8);
        static const unsigned int i5 = i4 | (i4 >> 16);
        
        static const unsigned int value = i5 + 1;
};

/*! \brief A dummy class needed to represent a 128 bit int */
class Packed16Bytes
{
public:
        typedef long long unsigned int type;

public:
        type a;
        type b;

public:
        __host__ __device__ Packed16Bytes(const Packed16Bytes& r) : a(r.a), b(r.b)
        {

        }

        __host__ __device__ Packed16Bytes() : a(0), b(0)
        {
        }

        __host__ __device__ Packed16Bytes(long long unsigned int i) : a(i), b(0)
        {
        }

	__host__ __device__ const Packed16Bytes operator~() const
	{
		Packed16Bytes result;
		result.a = ~a;
		result.b = ~b;
		return result;
	}

        __host__ __device__ const Packed16Bytes operator>>(
                unsigned int amount) const
        {
                if(amount < sizeof(type) * 8)
                {
                        type mask = ((type)1 << amount) - 1;
                        unsigned int reverseShift = sizeof(type) * 8 - amount;
         
                        type tmp1 = (a >> amount);
                        type tmp2 = 0;
                        if(reverseShift < 64)
                                tmp2 = (b & mask) << reverseShift;
       
                        Packed16Bytes result;
                
                        result.a = tmp1 | tmp2;
                        result.b = b >> amount;
                
                        return result;
                }
                else
                {
                        amount -= sizeof(type) * 8;
                        
                        Packed16Bytes result;
                
                        result.a = b >> amount;
                        result.b = 0;
                
                        return result;                  
                }
        }

        __host__ __device__ const Packed16Bytes operator<<(
                unsigned int amount) const
        {
                if(amount < sizeof(type) * 8)
                {
                        unsigned int reverseShift = sizeof(type) * 8 - amount;
                
                        type tmp1 = (b << amount);
                        type tmp2 = 0;

                        if(reverseShift < 64)
                                tmp2 = a >> reverseShift;

                        Packed16Bytes result;
                
                        result.a = (a << amount);
                        result.b = tmp1 | tmp2;
                
                        return result;
                }
                else
                {
                        amount -= sizeof(type) * 8;
                        
                        Packed16Bytes result;
                
                        result.b = a << amount;
                        result.a = 0;
                
                        return result;                  
                }
        }

        __host__ __device__ const Packed16Bytes operator-(
                const Packed16Bytes& value) const
        {
                Packed16Bytes result;
                
                result.b = b - value.b;
                result.a = a - value.a;
                
                if(result.a > a)
                {
                        --result.b;
                }

                return result;
        }

        __host__ __device__ bool operator<(const Packed16Bytes& value) const
        {
                if(b < value.b)
                {
                        return true;
                }
                else if(b == value.b && a < value.a)
                {
                        return true;
                }
                
                return false;
        }

        __host__ __device__ bool operator<(const unsigned int& value) const
        {
                if(b > 0)
                {
                        return false;
                }
                else if(a < value)
                {
                        return true;
                }
                
                return false;
        }

        __host__ __device__ const Packed16Bytes operator&(
                const Packed16Bytes& value) const
        {
                Packed16Bytes result;
                
                result.a = a & value.a;
                result.b = b & value.b;
                
                return result;
        }

        __host__ __device__ const Packed16Bytes operator|(
                const Packed16Bytes& value) const
        {
                Packed16Bytes result;
                
                result.a = a | value.a;
                result.b = b | value.b;
                
                return result;
        }

        __host__ __device__ operator long long unsigned int() const
        {
                return a;
        }
};

/*! \brief Get a basic type with the specified number of bits */
template<unsigned int bits>
class BasicTypeWithBits
{
public:
        typedef Packed16Bytes type;
};

template<>
class BasicTypeWithBits<8>
{
public:
        typedef unsigned char type;
};

template<>
class BasicTypeWithBits<16>
{
public:
        typedef unsigned short type;
};

template<>
class BasicTypeWithBits<32>
{
public:
        typedef unsigned int type;
};

template<>
class BasicTypeWithBits<64>
{
public:
        typedef long long unsigned int type;
};

template<unsigned int index, typename Tuple>
class BitsAt
{
public:
        static const unsigned int value = (index == 0) ? Tuple::bits0 : (index == 1)
                ? Tuple::bits1 : (index == 2) ? Tuple::bits2 :
                (index == 3) ? Tuple::bits3 : Tuple::bits4;
};
template<unsigned int index, typename Tuple>
class BitsUpTo
{
public:
        static const unsigned int value = BitsUpTo<index - 1, Tuple>::value
                + BitsAt<index - 1, Tuple>::value;
};

template<typename Tuple>
class BitsUpTo<1, Tuple>
{
public:
        static const unsigned int value = Tuple::bits0;
};

template<typename Tuple>
class BitsUpTo<0, Tuple>
{
public:
        static const unsigned int value = 0;
};

template<unsigned int index, typename Tuple>
class BitsAfter
{
public:
        static const unsigned int value = Tuple::bits
                - BitsUpTo<index, Tuple>::value
                - BitsAt<index, Tuple>::value;
};


template<bool condition, unsigned int IfTrue, unsigned int IfFalse>
class SelectIf
{
public:
        static const unsigned int value = 0;
};

/*! \brief A tuple descriptor contains the number of bits per element */
template<unsigned int b0, unsigned int b1 = 0, unsigned int b2 = 0,
        unsigned int b3 = 0, unsigned int b4 = 0>
class Tuple
{
public:
        typedef Tuple<b0, b1, b2, b3, b4> type;

public:
        static const unsigned int fields = 5;
        static const unsigned int bits0  = b0;
        static const unsigned int bits1  = b1;
        static const unsigned int bits2  = b2;
        static const unsigned int bits3  = b3;
        static const unsigned int bits4  = b4;
        static const unsigned int bits   = b0 + b1 + b2 + b3 + b4;

public:
        typedef typename BasicTypeWithBits<Clamp<
                b0 + b1 + b2 + b3>::value>::type BasicType;
};

template<typename T, unsigned int Index, typename Tuple>
__host__ __device__ T extract(const typename Tuple::BasicType& tuple)
{
        const unsigned int inRange = Index < 5;
        const unsigned int index   = Min<Index, 4>::value;
        const unsigned int shift   = BitsAfter<index, Tuple>::value;
        
        const typename Tuple::BasicType one = 1LLU;
        const typename Tuple::BasicType zero = 0LLU;
        
        const typename Tuple::BasicType mask = (one << BitsAt<index, Tuple>::value)
                - one;
        
        return inRange ? (tuple >> shift) & mask : zero;
}

template<unsigned int Index, typename Tuple>
__host__ __device__ double extract_double(const typename Tuple::BasicType& tuple)
{
        const unsigned int inRange = Index < 5;
        const unsigned int index   = Min<Index, 4>::value;
        const unsigned int shift   = BitsAfter<index, Tuple>::value;
        
        const typename Tuple::BasicType one = 1LLU;
        const typename Tuple::BasicType zero = 0LLU;
        
        const typename Tuple::BasicType mask = (one << BitsAt<index, Tuple>::value)
                - one;
        
        unsigned long long int tmp = inRange ? (tuple >> shift) & mask : zero;
	unsigned long long int *pointer_int = &tmp;
	double *pointer_double = (double *)pointer_int;
        return *pointer_double;
}

template<typename T, unsigned int Index, typename Tuple>
__host__ __device__ typename Tuple::BasicType insert(
        const typename Tuple::BasicType& tuple, const T& value)
{
        const unsigned int inRange = Index < 5;
        const unsigned int index   = Min<Index, 4>::value;
        const unsigned int shift   = BitsAfter<index, Tuple>::value;
        const typename Tuple::BasicType one = 1LLU;
        const typename Tuple::BasicType mask  = 
                ((one << BitsAt<index, Tuple>::value)
                - one) << shift;
	const typename Tuple::BasicType masked_tuple = tuple & (~mask);
        const typename Tuple::BasicType value_tmp = (typename Tuple::BasicType)value; 
        return inRange ? masked_tuple
                | ((value_tmp << shift) & mask) : tuple;
}

template<unsigned int Index, typename Tuple>
__host__ __device__ typename Tuple::BasicType insert_double(
        const typename Tuple::BasicType& tuple, const double& value)
{
	const double *pointer_double = &value;
	const unsigned long long int *pointer_int = (const unsigned long long int*)pointer_double;
        const unsigned int inRange = Index < 5;
        const unsigned int index   = Min<Index, 4>::value;
        const unsigned int shift   = BitsAfter<index, Tuple>::value;
        const typename Tuple::BasicType one = 1LLU;
        const typename Tuple::BasicType mask  = 
                ((one << BitsAt<index, Tuple>::value)
                - one) << shift;
	const typename Tuple::BasicType masked_tuple = tuple & (~mask);
        const typename Tuple::BasicType value_tmp = (typename Tuple::BasicType)(*pointer_int); 
        return inRange ? masked_tuple
                | ((value_tmp << shift) & mask) : tuple;
}

template<unsigned int Index, typename Tuple>
__host__ __device__ typename Tuple::BasicType insert_double2(
        const typename Tuple::BasicType& tuple, const double& value)
{
	const double *pointer_double = &value;
	const unsigned long long int *pointer_int = (const unsigned long long int*)pointer_double;
        const unsigned int inRange = Index < 5;
        const unsigned int index   = Min<Index, 4>::value;
        const unsigned int shift   = BitsAfter<index, Tuple>::value;
        const typename Tuple::BasicType one = 1LLU;
        const typename Tuple::BasicType mask  = 
                ((one << BitsAt<index, Tuple>::value)
                - one) << shift;
//	printf("shift %u\n", shift);
	const typename Tuple::BasicType masked_tuple = tuple & (~mask);
        const typename Tuple::BasicType value_tmp = (typename Tuple::BasicType)(*pointer_int); 
        return inRange ? masked_tuple
                | ((value_tmp << shift) & mask) : tuple;
}
//template<typename T, unsigned int Index, typename Tuple>
//__host__ __device__ typename Tuple::BasicType insert2(
//        const typename Tuple::BasicType& tuple, const T& value)
//{
//        const unsigned int inRange = Index < 5;
//        const unsigned int index   = Min<Index, 4>::value;
//        const unsigned int shift   = BitsAfter<index, Tuple>::value;
//        const typename Tuple::BasicType one = 1LLU;
//        const typename Tuple::BasicType tmp = (one << BitsAt<index, Tuple>::value);
//        const typename Tuple::BasicType mask  = 
//                (tmp
//                - one) << shift;
//	printf("tmp %llx %llx\n", tmp.a, tmp.b);
//	printf("mask %llx %llx\n", mask.a, mask.b);
//	const typename Tuple::BasicType masked_tuple = tuple & (~mask);
//        const typename Tuple::BasicType value_tmp = (typename Tuple::BasicType)value; 
//        return inRange ? masked_tuple
//                | ((value_tmp << shift) & mask) : tuple;
//}
template<typename Tuple, unsigned int fields>
__host__ __device__ typename Tuple::BasicType stripValues(
        typename Tuple::BasicType tuple)
{
        const bool inRange = fields < 5;
        const unsigned int shift = BitsAfter<fields, Tuple>::value;

        return inRange ? tuple >> shift : tuple;
}

template<typename Left, typename Right, typename Out, unsigned int fields>
__host__ __device__ typename Out::BasicType combine(
        typename Left::BasicType left, typename Right::BasicType right)
{
        typename Out::BasicType leftOut  = left;
        typename Out::BasicType rightOut = right;
        
        const unsigned int shift = Right::bits - BitsUpTo<fields + 1, Right>::value;
//      const typename Right::BasicType mask  = ((typename Right::BasicType)1 << shift)
//              - (typename Right::BasicType)1;
        const typename Out::BasicType mask  = ((typename Out::BasicType)1 << shift)
                - (typename Out::BasicType)1;

        return (leftOut << shift) | (rightOut & mask);
}

#endif

