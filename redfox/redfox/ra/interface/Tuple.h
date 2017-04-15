/*! \file Tuple.h
	\date Tuesday November 30, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The 
*/

#ifndef TUPLE_H_INCLUDED
#define TUPLE_H_INCLUDED

#include <stdio.h>

namespace ra
{
/*! \brief A namespace for the implementation of a tuple */
namespace tuple
{

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
		(v <= 8) ? 8 : (v <= 16) ? 16 : (v <= 32) ? 32 : (v <= 64) ? 64 : 
			(v <=128) ? 128 : (v <= 256) ? 256 : (v <= 512) ? 512 : v;
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

#if 0
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

	__host__ __device__ const Packed16Bytes operator>>(
		unsigned int amount) const
	{
		if(amount < sizeof(type) * 8)
		{
			type mask = ((type)1 << amount) - 1;
			unsigned int reverseShift = sizeof(type) * 8 - amount;
		
			Packed16Bytes result;
		
			result.a = (a >> amount) | ((b & mask) << reverseShift);
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
		
			Packed16Bytes result;
		
			result.a = (a << amount);
			result.b = (b << amount) | (a >> reverseShift);
		
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

	__host__ __device__ const Packed16Bytes operator&(
		const unsigned int& value) const
	{
		Packed16Bytes result;
		
		result.a = 0;
		result.b = b & value;
		
		return result;
	}

	__host__ __device__ const Packed16Bytes operator&(
		const unsigned long long int& value) const
	{
		Packed16Bytes result;
		
		result.a = 0;
		result.b = b & value;
		
		return result;
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
#endif

/*! \brief A dummy class needed to represent > 128 bit int */
template<int N>
class PackedNBytes
{
public:
	typedef long long unsigned int type;

public:
	type a[N];

public:
	__host__ __device__ PackedNBytes(const PackedNBytes& r)
	{
#pragma unroll
		for(int i = 0; i < N; ++i)
			a[i] = r.a[i];
	}

	__host__ __device__ PackedNBytes()
	{
#pragma unroll
		for(int i = 0; i < N; ++i)
			a[i] = 0;
	}

	__host__ __device__ PackedNBytes(long long unsigned int i)
	{
		a[0] = i;
		
#pragma unroll
		for(int i = 1; i < N; ++i)
			a[i] = 0;
	}
#if 0
	__host__ __device__ PackedNBytes(const Packed16Bytes& r)
	{
		a[0] = r.a;
		a[1] = r.b;

		for(int i = 2; i < N; ++i)
			a[i] = 0;
	}
#endif
	template<int M>
	__host__ __device__ PackedNBytes(const PackedNBytes<M>& r)
	{
		const int min = (N < M) ? N : M;
//		const int max = (N == min) ? M : N;

		#pragma unroll
		for(int i = 0; i < min; ++i)
			a[i] = r.a[i];

		#pragma unroll
//		for(int i = min; i < max; ++i)
		for(int i = min; i < N; ++i)
			a[i] = 0;
	}

	__host__ __device__ const PackedNBytes operator>>(
		unsigned int amount) const
	{
		const int bytes = amount / (sizeof(type) << 3);
		const int remain = amount - (sizeof(type) << 3) * bytes;

		PackedNBytes result;

		for(int i = N - 1; i > N - 1 - bytes; --i)
		{
			result.a[i] = 0;
		}

/*		if(remain == 0 & bytes < N)
		{
			for(int i = N - 1 - bytes; i >= 0; --i)
			{
				result.a[i] = a[i + bytes];
			}
		}
		else*/ if(bytes <= N - 1)
		{
			result.a[N - 1 - bytes] = a[N - 1] >> remain;

//			type mask = ((type)1 << remain) - 1;
			unsigned int reverseShift = (sizeof(type) << 3) - remain;

			for(int i = N - 2 - bytes; i >= 0; --i)
			{
				type tmp1 = (a[i + bytes] >> remain);
				type tmp2 = 0;
				if(reverseShift < 64)
					tmp2 = (a[i + bytes + 1]/* & mask*/) << reverseShift; 
				result.a[i] = tmp1 | tmp2; 

//				result.a[i] = (a[i + bytes] >> remain) 
//					| ((a[i + bytes + 1] & mask) << reverseShift);
			}
		}

		return result;
	}

	__host__ __device__ const PackedNBytes operator<<(
		unsigned int amount) const
	{
		int bytes = amount / (sizeof(type) << 3);
		int remain = amount % (sizeof(type) << 3);
		
//		printf("left shift input %llu, %llu\n", a[0], a[1]);
		PackedNBytes result;

		for(int i = 0; i < bytes; i++)
		{
			result.a[i] = 0;
		}


/*		if(remain == 0 && bytes < N)
		{
			for(int i = bytes; i < N; ++i)
			{
				result.a[i] = a[i - bytes];
			}
		}
		else*/ if(bytes <= N - 1)
		{
			result.a[bytes] = a[0] << remain;

//			type mask = ((type)1 << remain) - 1;
			unsigned int reverseShift = (sizeof(type) << 3)- remain;

			for(int i = bytes + 1; i < N; ++i)
			{
				//printf("%d\n", i);
				type tmp1 = (a[i - bytes] << remain);
				//printf("tmp1 %llx\n", tmp1);
				type tmp2 = 0;
				if(reverseShift < 64)
					tmp2 = (a[i - bytes - 1]/* & mask*/) >> reverseShift; 
				//printf("tmp2 %llx\n", tmp2);
				result.a[i] = tmp1 | tmp2; 
				//printf("result %llx\n", result.a[i]);
			}
		}
//		printf("left shift result %llu, %llu\n", result.a[0], result.a[1]);
		return result;
	}

	__host__ __device__ const PackedNBytes operator-(
		const PackedNBytes& value) const
	{
		PackedNBytes result;
//		printf("suba %llu, %llu\n", a[0], a[1]);	
//		printf("subb %llu, %llu\n", value.a[0], value.a[1]);
		unsigned long long int carryin = 0;
	
		#pragma unroll
		for(int i = 0; i < N; i++)
		{
			result.a[i] = a[i] - value.a[i] - carryin;
			
			if(a[i] < (value.a[i] + carryin))
				carryin = 1;
			else
				carryin = 0;	
		}
	
//		printf("sub result %llu, %llu\n", result.a[0], result.a[1]);	
		return result;
	}

        __host__ __device__ bool operator<(const unsigned long long int& value) const
        {
		#pragma unroll
		for(int i = 1; i < N; ++i)
			if(a[i] > 0) 
				return false;

        
                if(a[0] < value)
                {
                        return true;
                }

                return false;
        }

        __host__ __device__ bool operator<(const unsigned int& value) const
        {
		//#pragma unroll
		for(int i = 1; i < N; ++i)
			if(a[i] > 0) 
				return false;

        
                if(a[0] < value)
                {
                        return true;
                }

                return false;
        }

        __host__ __device__ bool operator<(const unsigned short& value) const
        {
		//#pragma unroll
		for(int i = 1; i < N; ++i)
			if(a[i] > 0) 
				return false;

        
                if(a[0] < value)
                {
                        return true;
                }

                return false;
        }


        __host__ __device__ bool operator<(const unsigned char& value) const
        {
		//#pragma unroll
		for(int i = 1; i < N; ++i)
			if(a[i] > 0) 
				return false;

        
                if(a[0] < value)
                {
                        return true;
                }

                return false;
        }

#if 0
	__host__ __device__ bool operator<(const Packed16Bytes& value) const
	{
		for(int i = N - 1; i > 1; --i)
		{
			if(a[i] > 0)
				return false;
		}

		if(a[1] < value.a)
			return true;	
	
		if(a[0] < value.b)
			return true;

		return false;
	}
#endif

	template<int M>
	__host__ __device__ bool operator<(const PackedNBytes<M>& value) const
	{
	//	printf("N %d M %d\n", N, M);
		if(N > M)
		{
		#pragma unroll
			for(int i = N - 1; i >= M; --i)
			{
				if(a[i] > 0)
					return false;
			}
			
		#pragma unroll
			for(int i = M - 1; i >= 0; --i)
			{
				if(a[i] < value.a[i])
					return true;
				else if(a[i] > value.a[i])
					return false;
				else
					continue;
			}
			
			return false;
		}
		else if(N == M)
		{
		#pragma unroll
			for(int i = N - 1; i >= 0; --i)
			{
				if(a[i] < value.a[i])
					return true;
				else if(a[i] > value.a[i])
					return false;
				else
					continue;
			}
			
			return false;
		}
		else
		{
//			printf("N < M\n");
		#pragma unroll
			for(int i = M - 1; i >= N; --i)
			{
				if(value.a[i] > 0)
					return false;
			}
			
		#pragma unroll
			for(int i = N - 1; i >= 0; --i)
			{
				if(a[i] < value.a[i])
				{
//					printf("return true %d\n", i);
					return true;
				}
				else if(a[i] > value.a[i])
				{
//					printf("return false %d \n", i);
					return false;
				}
				else
				{
//					printf("continue %d\n", i);
					continue;
				}
			}
			
			return false;
		}
	}

	__host__ __device__ const PackedNBytes operator&(
		const PackedNBytes& value) const
	{
		PackedNBytes result;

		//printf("anda %llx, %llx\n", a[0], a[1]);
		//printf("andb %llx, %llx\n", value.a[0], value.a[1]);
		#pragma unroll
		for(int i = 0; i < N; i++)
			result.a[i] = a[i] & value.a[i];	
		
		//printf("and result %llx, %llx\n", result.a[0], result.a[1]);
		return result;
	}

	__host__ __device__ const PackedNBytes operator~() const
	{
		PackedNBytes result;

		//printf("not input %llx, %llx\n", a[0], a[1]);
		#pragma unroll
		for(int i = 0; i < N; i++)
			result.a[i] = ~a[i];	
		
		//printf("not result %llx, %llx\n", result.a[0], result.a[1]);
		return result;
	}

	__host__ __device__ const PackedNBytes operator&(
		const unsigned long long int& value) const
	{
		PackedNBytes result;

		#pragma unroll
		for(int i = 1; i < N; i++)
			result.a[i] = 0;

		result.a[0] = a[0] & value;	
		
		return result;
	}

        __host__ __device__ const PackedNBytes operator|(
                const PackedNBytes& value) const
        {
                PackedNBytes result;

		#pragma unroll
		for(int i = 0; i < N; i++)
			result.a[i] = a[i] | value.a[i];	

                return result;
        }
	
	__host__ __device__ operator long long unsigned int() const
	{
//		printf("convert .....\n");
		return a[0];
	}

	__host__ __device__ operator char *() const
	{
//		printf("convert .....\n");
		return (char *)a[0];
	}

};

/*! \brief Get a basic type with the specified number of bits */
template<unsigned int bits>
class BasicTypeWithBits
{
public:
	typedef PackedNBytes<16> type;
};
template<>
class BasicTypeWithBits<512>
{
public:
	typedef PackedNBytes<8> type;
};
template<>
class BasicTypeWithBits<256>
{
public:
	typedef PackedNBytes<4> type;
};
template<>
class BasicTypeWithBits<128>
{
public:
	typedef PackedNBytes<2> type;
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
		(index == 3) ? Tuple::bits3 : (index == 4) ? Tuple::bits4 :
		(index == 5) ? Tuple::bits5 : (index == 6) ? Tuple::bits6 :
		(index == 7) ? Tuple::bits7 : (index == 8) ? Tuple::bits8 :
		(index == 9) ? Tuple::bits9 : (index == 10) ? Tuple::bits10 :
		(index == 11) ? Tuple::bits11 : (index == 12) ? Tuple::bits12 :
		(index == 13) ? Tuple::bits13 : (index == 14) ? Tuple::bits14 :
		Tuple::bits15;
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
	unsigned int b3 = 0, unsigned int b4 = 0, unsigned int b5 = 0,
	unsigned int b6 = 0, unsigned int b7 = 0, unsigned int b8 = 0,
	unsigned int b9 = 0, unsigned int b10 = 0, unsigned int b11 = 0,
	unsigned int b12 = 0, unsigned int b13 = 0, unsigned int b14 = 0,
	unsigned int b15 = 0>

class Tuple
{
public:
	typedef Tuple<b0, b1, b2, b3, b4, b5, b6, b7, 
		b8, b9, b10, b11, b12, b13, b14, b15> type;

public:
	static const unsigned int fields = 16;
	static const unsigned int bits0  = b0;
	static const unsigned int bits1  = b1;
	static const unsigned int bits2  = b2;
	static const unsigned int bits3  = b3;
	static const unsigned int bits4  = b4;
	static const unsigned int bits5  = b5;
	static const unsigned int bits6  = b6;
	static const unsigned int bits7  = b7;
	static const unsigned int bits8  = b8;
	static const unsigned int bits9  = b9;
	static const unsigned int bits10  = b10;
	static const unsigned int bits11 = b11;
	static const unsigned int bits12  = b12;
	static const unsigned int bits13  = b13;
	static const unsigned int bits14  = b14;
	static const unsigned int bits15  = b15;
	static const unsigned int bits   = b0 + b1 + b2 + b3 + b4 + b5 
		+ b6 + b7 + b8 + b9 + b10 + b11 + b12 + b13 + b14 + b15;

public:
	typedef typename BasicTypeWithBits<Clamp<
		b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 + b11 
		+ b12 + b13 + b14 + b15>::value>::type BasicType;
};

template<typename T, unsigned int Index, typename Tuple>
__host__ __device__ T extract(const typename Tuple::BasicType& tuple)
{
	const unsigned int inRange = Index < 16;
	const unsigned int index   = Min<Index, 15>::value;
	const unsigned int shift   = BitsAfter<index, Tuple>::value;
	
	const typename Tuple::BasicType one = 1LLU;
	const typename Tuple::BasicType zero = 0LLU;
	//printf("extract calculate mask: shift %u\n", shift);	
	const typename Tuple::BasicType mask = (one << BitsAt<index, Tuple>::value)
		- one;

	return inRange ? (tuple >> shift) & mask : zero;
}

//template<typename T, unsigned int Index, typename Tuple>
//__host__ __device__ T extract2(const typename Tuple::BasicType& tuple)
//{
//	const unsigned int inRange = Index < 16;
//	const unsigned int index   = Min<Index, 15>::value;
//	const unsigned int shift   = BitsAfter<index, Tuple>::value;
//	
//	const typename Tuple::BasicType one = 1LLU;
//	const typename Tuple::BasicType zero = 0LLU;
//	printf("extract calculate mask: shift %u\n", shift);	
//	const typename Tuple::BasicType mask = (one << BitsAt<index, Tuple>::value)
//		- one;
//	const unsigned long long int tmp = (tuple >> shift) & mask;
//	printf("extract %llx\n", tmp);
////	return tmp; 
//	return inRange ? (tuple >> shift) & mask : zero;
//}

template<unsigned int Index, typename Tuple>
__host__ __device__ double extract_double(const typename Tuple::BasicType& tuple)
{
        const unsigned int inRange = Index < 16;
        const unsigned int index   = Min<Index, 15>::value;
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

//template<unsigned int Index, typename Tuple>
//__host__ __device__ double extract_double2(const typename Tuple::BasicType& tuple)
//{
//        const unsigned int inRange = Index < 16;
//        const unsigned int index   = Min<Index, 15>::value;
//        const unsigned int shift   = BitsAfter<index, Tuple>::value;
//
//        const typename Tuple::BasicType one = 1LLU;
//        const typename Tuple::BasicType zero = 0LLU;
//
//        const typename Tuple::BasicType mask = (one << BitsAt<index, Tuple>::value)
//                - one;
//
//        unsigned long long int tmp = inRange ? (tuple >> shift) & mask : zero;
//        unsigned long long int *pointer_int = &tmp;
//        double *pointer_double = (double *)pointer_int;
//        return *pointer_double;
//}

template<typename T, unsigned int Index, typename Tuple>
__host__ __device__ typename Tuple::BasicType insert(
	const typename Tuple::BasicType& tuple, const T& value)
{
	const unsigned int inRange = Index < 16;
	const unsigned int index   = Min<Index, 15>::value;
	const unsigned int shift   = BitsAfter<index, Tuple>::value;
	//printf("insert %u %u\n", shift, BitsAt<index, Tuple>::value);
	const typename Tuple::BasicType mask  = 
		(((typename Tuple::BasicType)1 << BitsAt<index, Tuple>::value)
		- (typename Tuple::BasicType)1) << shift;
	return inRange ? (tuple & ~mask)
		| ((((typename Tuple::BasicType) value) << shift) & mask) : tuple;
}

//template<typename T, unsigned int Index, typename Tuple>
//__host__ __device__ typename Tuple::BasicType insert2(
//	const typename Tuple::BasicType& tuple, const T& value)
//{
//	printf("%u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\n", Tuple::bits0, Tuple::bits1, Tuple::bits2, Tuple::bits3, Tuple::bits4, Tuple::bits5,  Tuple::bits6, Tuple::bits7, Tuple::bits8, Tuple::bits9, Tuple::bits10, Tuple::bits11, Tuple::bits12, Tuple::bits13, Tuple::bits14, Tuple::bits15);
//	const unsigned int inRange = Index < 16;
//	printf("1 inRange %u\n", inRange);
//	const unsigned int index   = Min<Index, 15>::value;
//	printf("2 index %u\n", index);
//	const unsigned int shift   = BitsAfter<index, Tuple>::value;
//	printf("3 shift %u bitsat %u\n", shift, BitsAt<index, Tuple>::value);
//	//printf("insert %u %u\n", shift, BitsAt<index, Tuple>::value);
//	const typename Tuple::BasicType mask  = 
//		(((typename Tuple::BasicType)1 << BitsAt<index, Tuple>::value)
//		- (typename Tuple::BasicType)1) << shift;
//	printf("4\n");
//	return inRange ? (tuple & ~mask)
//		| ((((typename Tuple::BasicType) value) << shift) & mask) : tuple;
//}

template<unsigned int Index, typename Tuple>
__host__ __device__ typename Tuple::BasicType insert_double(
        const typename Tuple::BasicType& tuple, const double& value)
{
        const double *pointer_double = &value;
        const unsigned long long int *pointer_int = (const unsigned long long int*)pointer_double;
        const unsigned int inRange = Index < 16;
        const unsigned int index   = Min<Index, 15>::value;
        const unsigned int shift   = BitsAfter<index, Tuple>::value;
	const typename Tuple::BasicType mask  = 
		(((typename Tuple::BasicType)1 << BitsAt<index, Tuple>::value)
		- (typename Tuple::BasicType)1) << shift;

	return inRange ? (tuple & ~mask)
		| ((((typename Tuple::BasicType) (*pointer_int)) << shift) & mask) : tuple;
}

template<typename Tuple, unsigned int fields>
__host__ __device__ typename Tuple::BasicType stripValues(
	typename Tuple::BasicType tuple)
{
	const bool inRange = fields < 16;
	const unsigned int shift = BitsAfter<fields - 1, Tuple>::value;
	return inRange ? tuple >> shift : tuple;
}

template<typename Tuple, unsigned int fields>
__host__ __device__ typename Tuple::BasicType stripKeys(
	typename Tuple::BasicType tuple)
{
	const bool inRange = fields < 16;
	const unsigned int shift = BitsAfter<fields - 1, Tuple>::value;
	const typename Tuple::BasicType mask  = 
		(((typename Tuple::BasicType)1 << shift)
		- (typename Tuple::BasicType)1);

	return inRange ? tuple & mask : tuple;
}

template<typename Tuple, unsigned int fields>
__host__ __device__ typename Tuple::BasicType stripKeys2(
	typename Tuple::BasicType tuple)
{
	const bool inRange = fields < 16;
	const unsigned int shift = BitsAfter<fields - 1, Tuple>::value;
	const typename Tuple::BasicType mask  = 
		(((typename Tuple::BasicType)1 << shift)
		- (typename Tuple::BasicType)1);
	typename Tuple::BasicType tmp = tuple & mask;
//if(threadIdx.x == 0 && blockIdx.x == 0) printf("strip key %llx %llx %llx %llx %llx\n", tmp.a[0], tmp.a[1], tmp.a[2], tmp.a[3], tmp.a[4]);
	return inRange ? tuple & mask : tuple;
}

template<typename Tuple, unsigned int fields>
__host__ __device__ typename Tuple::BasicType restoreValues(
	typename Tuple::BasicType tuple)
{
	const bool inRange = fields < 16;
	const unsigned int shift = BitsAfter<fields - 1, Tuple>::value;

	return inRange ? tuple << shift : tuple;
}

template<typename Left, typename Right, typename Out, unsigned int fields>
__host__ __device__ typename Out::BasicType combine(
	typename Left::BasicType left, typename Right::BasicType right)
{
	typename Out::BasicType leftOut  = left;
	typename Out::BasicType rightOut = right;
	
	//const unsigned int shift = BitsAfter<fields - 1, Right>::value;
	const unsigned int shift = Right::bits - BitsUpTo<fields, Right>::value;
	const typename Out::BasicType mask  = ((typename Out::BasicType)1 << shift)
		- (typename Out::BasicType)1;
	
	return (leftOut << shift) | (rightOut & mask);
}
//template<typename Left, typename Right, typename Out, unsigned int fields>
//__host__ __device__ typename Out::BasicType combine2(
//	typename Left::BasicType left, typename Right::BasicType right)
//{
//	typename Out::BasicType leftOut  = left;
//	typename Out::BasicType rightOut = right;
//	
//	//const unsigned int shift = BitsAfter<fields - 1, Right>::value;
//	const unsigned int shift = Right::bits - BitsUpTo<fields, Right>::value;
////printf("combine shift %u\n", shift);
//	const typename Out::BasicType mask  = ((typename Out::BasicType)1 << shift)
//		- (typename Out::BasicType)1;
////typename Out::BasicType	temp = (typename Out::BasicType)1 << shift;
////printf("combine mask %llx %llx %llx %llx %llx %llx %llx\n", mask.a[0], mask.a[1], mask.a[2], mask.a[3], mask.a[4], mask.a[5], mask.a[6]);
////printf("combine temp %llx %llx %llx %llx %llx %llx %llx\n", temp.a[0], temp.a[1], temp.a[2], temp.a[3], temp.a[4], temp.a[5], temp.a[6]);
//	return (leftOut << shift) | (rightOut & mask);
//}
}

}

#endif

