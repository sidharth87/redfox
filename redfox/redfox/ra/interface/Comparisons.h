/*! \file Comparisons.h
	\date Friday June 5, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for Datalog comparison functors.
*/

#ifndef COMPARISONS_H_INCLUDED
#define COMPARISONS_H_INCLUDED

namespace ra
{
	static __host__ __device__ bool like_string(const char *string1, const char *string2, int string2_size, int *string1_new_pos)
	{
	        int i = 0;
	        int j = 0;
	        while(string1[i] != '\0')
	        {
	                if(string1[i] == string2[j])
	                {
	                        j++;
	
	                        int ii = i + 1;
	
	                        while(string1[ii] != '\0' && j < string2_size)
	                        {
	                                if (string1[ii] == string2[j])
	                                {
	                                        ii++;
	                                        j++;
	                                }
	                                else
	                                {
	                                        break;
	                                }
	                        }
	
	                        if (j == string2_size)
	                        {
	                                *string1_new_pos = ii;
	                                return true;
	                        }
	                }
	
	                j = 0;
	                i++;
	        }
	
	        return false;
	}
 
	static __host__ __device__ bool checkhead(const char *string1, const char *string2, int *string1_start, int *string2_start)
	{
	        int i = 0;
	
	        while(string1[i] != '\0' && string2[i] != '\0' && string2[i] != '%')
	        {
	                if(string1[i] != string2[i])
	                        return false;
	
	                i++;
	        }
	
	        *string1_start = i;
	        *string2_start = i;
	
	        if(string2[i] == '\0' || string2[i] == '%')
	                return true;
	        else
	                return false;
	
	}

	static __host__ __device__ bool checktail(const char *string1, const char *string2, int string1_length, int string2_length, int string1_start, int string2_start, int *string1_end, int *string2_end)
	{
	        int i = string1_length;
	        int j = string2_length;
	
	        while(i > string1_start && j > string2_start)
	        {
	                i--;
	                j--;
	
	                if(string2[j] == '%')
	                        break;
	
	                if(string1[i] != string2[j])
	                        return false;
	
	        }
	
	        *string1_end = i + 1;
	        *string2_end = j + 1;
	
	        if(string2[j] == '%')
	                return true;
	        else
	                return false;
	
	}

	static __host__ __device__ int stringlength(const char *string)
	{
	        int i = 0;
	
	        while(string[i++] != '\0');
	
	        return (i - 1);
	}

	/*! \brief A namespace for GPU datalog comparison functors */
	namespace comparisons
	{
		/*! \brief The eq operator */
		template< typename T >
		class eq
		{
			public:
				typedef T value_type;
			
			public:
				__host__ __device__ bool operator()( const T& one,
					const T& two ) const
				{
					return one == two;
				}
		};
		
		/*! \brief The ne operator */
		template< typename T >
		class ne
		{
			public:
				typedef T value_type;
			
			public:
				__host__ __device__ bool operator()( const T& one,
					const T& two ) const
				{
					return one != two;
				}
		};
		
		/*! \brief The lt operator */
		template< typename T >
		class lt
		{
			public:
				typedef T value_type;
			
			public:
				__host__ __device__ bool operator()( const T& one,
					const T& two ) const
				{
					return one < two;
				}
		};
	
		class ltdouble
		{
			public:
				typedef double T;
			
			public:
				__host__ __device__ bool operator()( const T& one,
					const T& two ) const
				{
					return( two - one) > 1e-6;
				}
		};
	
		class ledouble
		{
			public:
				typedef double T;
			
			public:
				__host__ __device__ bool operator()( const T& one,
					const T& two ) const
				{
					return( two - one) > -1e-6;
				}
		};
	
		class gtdouble
		{
			public:
				typedef double T;
			
			public:
				__host__ __device__ bool operator()( const T& one,
					const T& two ) const
				{
					return (one - two) > 1e-6;
				}
		};
		
		class gedouble
		{
			public:
				typedef double T;
			
			public:
				__host__ __device__ bool operator()( const T& one,
					const T& two ) const
				{
					return (one - two) > -1e-6;
				}
		};

		/*! \brief The gt operator */
		template< typename T >
		class sgt
		{
			public:
				typedef T value_type;
			
			public:
				__host__ __device__ bool operator()( const T& one,
					const T& two ) const
				{
					return one > two;
				}
		};
		
		/*! \brief The gt operator */
		template< typename T >
		class gt
		{
			public:
				typedef T value_type;
			
			public:
				__host__ __device__ bool operator()( const T& one,
					const T& two ) const
				{
					return one > two;
				}
		};
		
		/*! \brief The le operator */
		template< typename T >
		class le
		{
			public:
				typedef T value_type;
			
			public:
				__host__ __device__ bool operator()( const T& one,
					const T& two ) const
				{
					return one <= two;
				}
		};
	
		/*! \brief The le operator */
		template< typename T >
		class sle
		{
			public:
				typedef T value_type;
			
			public:
				__host__ __device__ bool operator()( const T& one,
					const T& two ) const
				{
					return one <= two;
				}
		};
		
		/*! \brief The ge operator */
		template< typename T >
		class ge
		{
			public:
				typedef T value_type;
			
			public:
				__host__ __device__ bool operator()( const T& one,
					const T& two ) const
				{
					return one >= two;
				}
		};
		
		/*! \brief The ge operator */
		template< typename T >
		class sge
		{
			public:
				typedef T value_type;
			
			public:
				__host__ __device__ bool operator()( const T& one,
					const T& two ) const
				{
					return one >= two;
				}
		};



		template< typename T >
		class like 
		{
			public:
				typedef T value_type;
			
			public:
				__host__ __device__ bool operator()( const T& one,
					const T& two ) const
				{
				        const char *string1 = (char *)one;
				        const char *string2 = (char *)two;
				
				        int string1_start = 0;
				        int string2_start = 0;
				
				        if(string2[0] != '%')
				                if(!checkhead(string1, string2, &string1_start, &string2_start))
				                        return false;
				
				        if(string2[string2_start] == '\0')
				        {
				                if(string1[string1_start] == '\0')
				                        return true;
				                else
				                        return false;
				        }
				
				        int string1_length = stringlength(string1);
				        int string2_length = stringlength(string2);
				        int string1_end = string1_length;
				        int string2_end = string2_length;
				
				        if(string2[string2_length - 1] != '%')
				                if(!checktail(string1, string2, string1_length, string2_length, string1_start, string2_start, &string1_end, &string2_end))
				                        return false;
				
				        int i = string1_start;
				        int j = string2_start;
				        int string1_pos = 0;
				        const char *word_start = NULL;
				        const char *word_end = NULL;
				
				        while(j < string2_end)
				        {
				                while(string2[j] == '%')
				                        j++;
				
				                if(j >= string2_end)
				                        break;
				                else
				                {
				                        word_start = string2 + j;
				
				                        while(j < string2_end && string2[j] != '%')
				                                j++;
				
				                        word_end = string2 + j;
				
				                        if(like_string(string1 + i, word_start, word_end - word_start, &string1_pos))
				                                i += string1_pos;
				                        else
				                                return false;
				                }
				        }
				
				        if(j == string2_end)
				                return true;
				        else
				                return false;
				}
		};	
		template< typename T >
		class notlike 
		{
			public:
				typedef T value_type;
			
			public:
				__host__ __device__ bool operator()( const T& one,
					const T& two ) const
				{
				        const char *string1 = (char *)one;
				        const char *string2 = (char *)two;
				
				        int string1_start = 0;
				        int string2_start = 0;
				
				        if(string2[0] != '%')
				                if(!checkhead(string1, string2, &string1_start, &string2_start))
				                        return true;
				
				        if(string2[string2_start] == '\0')
				        {
				                if(string1[string1_start] == '\0')
				                        return false;
				                else
				                        return true;
				        }
				
				        int string1_length = stringlength(string1);
				        int string2_length = stringlength(string2);
				        int string1_end = string1_length;
				        int string2_end = string2_length;
				
				        if(string2[string2_length - 1] != '%')
				                if(!checktail(string1, string2, string1_length, string2_length, string1_start, string2_start, &string1_end, &string2_end))
				                        return true;
				
				        int i = string1_start;
				        int j = string2_start;
				        int string1_pos = 0;
				        const char *word_start = NULL;
				        const char *word_end = NULL;
				
				        while(j < string2_end)
				        {
				                while(string2[j] == '%')
				                        j++;
				
				                if(j >= string2_end)
				                        break;
				                else
				                {
				                        word_start = string2 + j;
				
				                        while(j < string2_end && string2[j] != '%')
				                                j++;
				
				                        word_end = string2 + j;
				
				                        if(like_string(string1 + i, word_start, word_end - word_start, &string1_pos))
				                                i += string1_pos;
				                        else
				                                return true;
				                }
				        }
				
				        if(j == string2_end)
				                return false;
				        else
				                return true;

				}
		};
		template< typename T >
		class eqstring 
		{
			public:
				typedef T value_type;
			
			public:
				__host__ __device__ bool operator()( const T& one,
					const T& two ) const
				{
					const char *string1 = (char *) one;
					const char *string2 = (char *) two;
		
					int i = 0;

					while(string1[i] != '\0' && string2[i] != '\0')
					{
						if(string1[i] != string2[i])
							return false;

						i++;
					}

					if(string1[i] == '\0' && string2[i] == '\0')
						return true;
					else
						return false;
				}
		};
		/*! \brief Convert a binary operator to a unary operator */
		template< typename T >
		class unary
		{
			public:
				typedef typename T::value_type value_type;
			
			public:
				value_type value;
			
			public:
				__host__ __device__ bool operator()( const value_type& one )
				{
					return T()( one, value );
				}
		};
		
		/*! \brief Convert a binary operator to a unary operator on the value */
		template< typename T, typename Compare >
		class CompareValue
		{
			public:
				typedef T value_type;
				typedef typename T::first_type first_type;
				typedef typename T::second_type second_type;
				
			public:
				Compare comp;
				second_type value;
			
			public:
				CompareValue( const second_type& v = second_type(), 
					const Compare& c = Compare() ) : comp( c ), value( v )
				{
				
				}
			
				bool operator()( const value_type& one )
				{
					return comp( one.second, value );
				}
		};
		
	}
}

#endif

