/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : UART4 라인 명령
  * - 'g' 수신: 주행 시작(RUN) + "GO\n" (전진)
  * - 'b' 수신: 주행 시작(RUN) + "BACK\n" (후진) [추가됨]
  * - 't' 수신: 정지(IDLE) + "OK\n"
  * - 'status' 수신: "RUN\n" 또는 "IDLE\n"
  * 부팅 시에는 IDLE(정지)로 시작.
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>

/* ================= User Config ================= */
#define BAUDRATE     115200         // UART4, Pi와 동일
#define RX_LINE_MAX  64

/* [수정] 직진성 보정을 위한 속도 분리 */
/* 왼쪽으로 치우치므로 왼쪽 바퀴 출력을 높임 */
#define SPEED_L      355            // FL, RL (왼쪽)
#define SPEED_R      345            // FR, RR (오른쪽)

/* ================= HAL Handles ================= */
TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim3;
TIM_HandleTypeDef htim4;
TIM_HandleTypeDef htim8;
TIM_HandleTypeDef htim9;
UART_HandleTypeDef huart4;

/* ================= App State ================= */
static uint8_t  rx_byte;              // UART4 1-byte buffer
static char     rx_line[RX_LINE_MAX]; // line buffer
static uint32_t rx_len = 0;
static uint8_t  line_ready = 0;

static uint8_t  is_running = 0;       // 0=IDLE(정지), 1=RUN(주행)

/* ================= Prototypes ================= */
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM1_Init(void);
static void MX_TIM3_Init(void);
static void MX_TIM4_Init(void);
static void MX_TIM8_Init(void);
static void MX_TIM9_Init(void);
static void MX_UART4_Init(void);
static void UART4_GPIO_Init(void);

static void append_char_to_line(uint8_t c);
static void send4(const char* s);

void WheelSpeeds(int fl, int fr, int rl, int rr);
void Robot_Forward(void);
void Robot_Backward(void); // [추가] 후진 함수 프로토타입
void Robot_Stop(void);

/* ================= Main ================= */
int main(void)
{
  HAL_Init();
  SystemClock_Config();

  MX_GPIO_Init();
  MX_TIM1_Init();
  MX_TIM3_Init();
  MX_TIM4_Init();
  MX_TIM8_Init();
  MX_TIM9_Init();
  MX_UART4_Init();

  /* PWM 시작 */
  HAL_TIM_PWM_Start(&htim8, TIM_CHANNEL_1); // RL
  HAL_TIM_PWM_Start(&htim8, TIM_CHANNEL_2); // FL
  HAL_TIM_PWM_Start(&htim9, TIM_CHANNEL_1); // FR
  HAL_TIM_PWM_Start(&htim9, TIM_CHANNEL_2); // RR

  /* UART4 수신 인터럽트 무장 */
  HAL_UART_Receive_IT(&huart4, &rx_byte, 1);

  /* 부팅 시: IDLE(정지)로 시작 */
  is_running = 0;
  Robot_Stop();
  send4("============================\n");
  send4("[STM] READY 115200\n");
  send4("[STM] IDLE (waiting 'g' or 'b')\n");
  send4("============================\n");

  while (1)
  {
    if (line_ready)
    {
      line_ready = 0;

      // 소문자/트림
      char cmd[RX_LINE_MAX];
      strncpy(cmd, rx_line, RX_LINE_MAX - 1);
      cmd[RX_LINE_MAX - 1] = '\0';

      size_t n = strlen(cmd);
      while (n && (cmd[n - 1] == '\r' || cmd[n - 1] == '\n' || isspace((unsigned char)cmd[n - 1]))) cmd[--n] = '\0';
      size_t i = 0; while (cmd[i] && isspace((unsigned char)cmd[i])) i++;
      if (i) memmove(cmd, cmd + i, strlen(cmd + i) + 1);
      for (char *p = cmd; *p; ++p) *p = (char)tolower((unsigned char)*p);

      if (strcmp(cmd, "t") == 0)
      {
        // 정지(IDLE)
        is_running = 0;
        Robot_Stop();
        send4("OK\n");
      }
      else if (strcmp(cmd, "g") == 0)
      {
        // 전진 (GO)
        is_running = 1;
        Robot_Forward();
        send4("GO\n");
      }
      else if (strcmp(cmd, "b") == 0)
      {
        // [추가] 후진 (BACK)
        is_running = 1;
        Robot_Backward();
        send4("BACK\n");
      }
      else if (strcmp(cmd, "status") == 0)
      {
        send4(is_running ? "RUN\n" : "IDLE\n");
      }

      rx_len = 0;
      rx_line[0] = '\0';
    }
  }
}

/* ================= UART Callbacks ================= */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == UART4)
  {
    append_char_to_line(rx_byte);
    HAL_UART_Receive_IT(&huart4, &rx_byte, 1);
  }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == UART4)
  {
    volatile uint32_t sr = huart->Instance->SR;
    volatile uint32_t dr = huart->Instance->DR;
    (void)sr; (void)dr;
    rx_len = 0;
    HAL_UART_Receive_IT(&huart4, &rx_byte, 1);
  }
}

/* ================= Line Buffer ================= */
static void append_char_to_line(uint8_t c)
{
  // 개행 수신 시 라인 종료 처리
  if (c == '\n' || c == '\r')
  {
    if (rx_len > 0)
    {
      rx_line[rx_len] = '\0';
      line_ready = 1;
    }
    return;
  }

  // 가시문자 범위만 수용
  if (c < 0x20 || c > 0x7E) return;

  if (rx_len < RX_LINE_MAX - 1)
  {
    rx_line[rx_len++] = (char)c;
    rx_line[rx_len]   = '\0';
  }
  else
  {
    rx_len = 0;
    rx_line[0] = '\0';
  }
}

/* ================= UART utils ================= */
static void send4(const char* s)
{
  HAL_UART_Transmit(&huart4, (uint8_t*)s, (uint16_t)strlen(s), 100);
}

/* ================= Motor Control ================= */
void WheelSpeeds(int fl_speed, int fr_speed, int rl_speed, int rr_speed)
{
  int pwm_fl = abs(fl_speed);
  int pwm_fr = abs(fr_speed);
  int pwm_rl = abs(rl_speed);
  int pwm_rr = abs(rr_speed);

  if (pwm_fl > 999) pwm_fl = 999;
  if (pwm_fr > 999) pwm_fr = 999;
  if (pwm_rl > 999) pwm_rl = 999;
  if (pwm_rr > 999) pwm_rr = 999;

  /* 방향핀 (검증된 핀맵) - 속도가 양수면 정방향, 음수면 역방향 */
  // FR (PE12, PE13)
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_12, fr_speed > 0 ? GPIO_PIN_SET : GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_13, fr_speed > 0 ? GPIO_PIN_RESET : GPIO_PIN_SET);

  // RL (PA2, PA3)
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_2, rl_speed > 0 ? GPIO_PIN_SET : GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_3, rl_speed > 0 ? GPIO_PIN_RESET : GPIO_PIN_SET);

  // FL (PA4, PA5)
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, fl_speed > 0 ? GPIO_PIN_RESET : GPIO_PIN_SET);
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, fl_speed > 0 ? GPIO_PIN_SET : GPIO_PIN_RESET);

  // RR (PE14, PE15)
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_14, rr_speed > 0 ? GPIO_PIN_SET : GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_15, rr_speed > 0 ? GPIO_PIN_RESET : GPIO_PIN_SET);

  /* PWM 출력 */
  __HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_2, pwm_fl); // FL
  __HAL_TIM_SET_COMPARE(&htim9, TIM_CHANNEL_1, pwm_fr); // FR
  __HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_1, pwm_rl); // RL
  __HAL_TIM_SET_COMPARE(&htim9, TIM_CHANNEL_2, pwm_rr); // RR
}

/* [직진] 보정값 적용 */
void Robot_Forward(void)
{
  WheelSpeeds(SPEED_L, SPEED_R, SPEED_L, SPEED_R);
}

/* [추가] [후진] 보정값에 음수 부호만 붙여서 전달 */
void Robot_Backward(void)
{
  WheelSpeeds(-SPEED_L, -SPEED_R, -SPEED_L, -SPEED_R);
}

void Robot_Stop(void)
{
  WheelSpeeds(0, 0, 0, 0);
}

/* ================= Clock / GPIO / Timer / UART Inits ================= */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 25;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) { Error_Handler(); }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                                RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;   // 168 MHz
  RCC_ClkInitStruct.AHBCLKDivider  = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK) { Error_Handler(); }
}

static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();

  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_2|GPIO_PIN_3|GPIO_PIN_4|GPIO_PIN_5, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_12|GPIO_PIN_13|GPIO_PIN_14|GPIO_PIN_15, GPIO_PIN_RESET);

  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;

  GPIO_InitStruct.Pin = GPIO_PIN_2|GPIO_PIN_3|GPIO_PIN_4|GPIO_PIN_5;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = GPIO_PIN_12|GPIO_PIN_13|GPIO_PIN_14|GPIO_PIN_15;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);
}

static void MX_TIM8_Init(void)
{
  TIM_OC_InitTypeDef sConfigOC = {0};
  TIM_BreakDeadTimeConfigTypeDef sBreak = {0};

  htim8.Instance = TIM8;
  htim8.Init.Prescaler = 9 - 1;
  htim8.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim8.Init.Period = 1000 - 1;
  htim8.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim8.Init.RepetitionCounter = 0;
  htim8.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim8) != HAL_OK) { Error_Handler(); }

  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;

  HAL_TIM_PWM_ConfigChannel(&htim8, &sConfigOC, TIM_CHANNEL_1);
  HAL_TIM_PWM_ConfigChannel(&htim8, &sConfigOC, TIM_CHANNEL_2);

  HAL_TIMEx_ConfigBreakDeadTime(&htim8, &sBreak);
  HAL_TIM_MspPostInit(&htim8);
}

static void MX_TIM9_Init(void)
{
  TIM_OC_InitTypeDef sConfigOC = {0};
  htim9.Instance = TIM9;
  htim9.Init.Prescaler = 9 - 1;
  htim9.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim9.Init.Period = 1000 - 1;
  htim9.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim9.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim9) != HAL_OK) { Error_Handler(); }

  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;

  HAL_TIM_PWM_ConfigChannel(&htim9, &sConfigOC, TIM_CHANNEL_1);
  HAL_TIM_PWM_ConfigChannel(&htim9, &sConfigOC, TIM_CHANNEL_2);
  HAL_TIM_MspPostInit(&htim9);
}

static void UART4_GPIO_Init(void)
{
  __HAL_RCC_GPIOA_CLK_ENABLE();
  GPIO_InitTypeDef G = {0};
  G.Pin = GPIO_PIN_0 | GPIO_PIN_1;  // PA0=TX, PA1=RX
  G.Mode = GPIO_MODE_AF_PP;
  G.Pull = GPIO_PULLUP;
  G.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  G.Alternate = GPIO_AF8_UART4;
  HAL_GPIO_Init(GPIOA, &G);
}

static void MX_UART4_Init(void)
{
  __HAL_RCC_UART4_CLK_ENABLE();
  UART4_GPIO_Init();

  huart4.Instance = UART4;
  huart4.Init.BaudRate = BAUDRATE;
  huart4.Init.WordLength = UART_WORDLENGTH_8B;
  huart4.Init.StopBits = UART_STOPBITS_1;
  huart4.Init.Parity = UART_PARITY_NONE;
  huart4.Init.Mode = UART_MODE_TX_RX;
  huart4.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart4.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart4) != HAL_OK) { Error_Handler(); }

  __HAL_UART_ENABLE_IT(&huart4, UART_IT_RXNE);
}

/* ================= Error Handler ================= */
void Error_Handler(void)
{
  __disable_irq();
  while (1) {}
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
  (void)file; (void)line;
}
#endif

/* ---- Dummy Timer Inits (not used in this version) ---- */
static void MX_TIM1_Init(void) { /* not used */ }
static void MX_TIM3_Init(void) { /* not used */ }
static void MX_TIM4_Init(void) { /* not used */ }
